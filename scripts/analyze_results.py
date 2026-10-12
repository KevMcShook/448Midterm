"""Create figures, uncertainty estimates, and the RQ4 style analysis."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import tempfile
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'448-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import torch
from torch.utils.data import DataLoader
from train_experiments import LABELS,TextCNN,TextLSTM,TextDataset,collate,tokenize,encode,evaluate

COLORS = ['#226F91','#C65D3B','#5D8B52']

def score(y,pred):
    return {'accuracy':float(accuracy_score(y,pred)),'macro_f1':float(f1_score(y,pred,labels=LABELS,average='macro',zero_division=0))}

def cluster_interval(predictions, repetitions=2000):
    label_ids = {label:i for i,label in enumerate(LABELS)}
    group_ids = {g:i for i,g in enumerate(sorted({r['split_group'] for r in predictions}))}
    matrices = np.zeros((len(group_ids),3,3),dtype=np.int64)
    for row in predictions:
        matrices[group_ids[row['split_group']],label_ids[row['true_label']],label_ids[row['predicted_label']]] += 1
    rng = np.random.default_rng(448)
    metrics = []
    for _ in range(repetitions):
        cm = matrices[rng.integers(len(matrices),size=len(matrices))].sum(axis=0)
        den = cm.sum(axis=0)+cm.sum(axis=1)
        f1 = np.divide(2*np.diag(cm),den,out=np.zeros(3),where=den!=0).mean()
        metrics.append([np.trace(cm)/cm.sum(),f1])
    low,high = np.quantile(metrics,[0.025,0.975],axis=0)
    return {'accuracy_95_interval':[float(low[0]),float(high[0])],'macro_f1_95_interval':[float(low[1]),float(high[1])],'bootstrap_repetitions':repetitions,'resampling_unit':'split_group'}

def style_features(text):
    words = re.findall(r'\b\w+\b',text.lower())
    lines = text.splitlines()
    n = max(1,len(words))
    return [np.log1p(len(words)),np.log1p(len(text)),len(set(words))/n,
            len(lines)/n, text.count('**')/n,text.count('```')/n,
            sum(bool(re.match(r'^\s*#{1,6}\s',line)) for line in lines)/n,
            sum(bool(re.match(r'^\s*(?:[-*+] |\d+[.)] )',line)) for line in lines)/n,
            text.count('!')/n,text.count('?')/n]

def paired_accuracy_interval(output_predictions, combined_predictions, repetitions=2000):
    combined = {r['example_id']:r for r in combined_predictions}
    groups = {}
    for row in output_predictions:
        other = combined[row['example_id']]
        assert row['true_label'] == other['true_label'] and row['split_group'] == other['split_group']
        group = groups.setdefault(row['split_group'],[0,0])
        group[0] += int(other['predicted_label']==row['true_label'])-int(row['predicted_label']==row['true_label'])
        group[1] += 1
    values = np.array(list(groups.values()))
    rng = np.random.default_rng(448)
    deltas = []
    for _ in range(repetitions):
        summed = values[rng.integers(len(values),size=len(values))].sum(axis=0)
        deltas.append(summed[0]/summed[1])
    return {'combined_minus_output_accuracy':float(values[:,0].sum()/values[:,1].sum()),
            'paired_group_bootstrap_95_interval':np.quantile(deltas,[.025,.975]).tolist()}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path('data/cleaned/claude_gemini_qwen_balanced.parquet'))
    parser.add_argument('--results',type=Path,default=Path('results'))
    args = parser.parse_args()
    threadpool_limits(2)
    torch.set_num_threads(2)
    frame = pd.read_parquet(args.data).reset_index(drop=True)
    manifest = pd.read_parquet(args.results/'split_manifest.parquet').set_index('example_id')
    frame['split'] = frame.example_id.map(manifest['split'])
    assert frame['split'].notna().all()
    train = frame['split'].eq('train').to_numpy()
    test = frame['split'].eq('test').to_numpy()
    metrics = json.loads((args.results/'all_metrics.json').read_text())
    intervals = {}
    for result in metrics:
        predictions = json.loads((args.results/'predictions'/f"{result['experiment']}.json").read_text())
        intervals[result['experiment']] = cluster_interval(predictions)
    (args.results/'uncertainty.json').write_text(json.dumps(intervals,indent=2)+'\n')
    deltas = {}
    for architecture in ('cnn','lstm'):
        paired = [json.loads((args.results/'predictions'/f'{architecture}_{mode}.json').read_text()) for mode in ('output','combined')]
        deltas[architecture] = paired_accuracy_interval(*paired)
    (args.results/'rq2_differences.json').write_text(json.dumps(deltas,indent=2)+'\n')

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})
    fig,axes = plt.subplots(1,2,figsize=(9.5,3.6))
    for axis,metric,title in zip(axes,['accuracy','macro_f1'],['Test accuracy','Test macro-F1']):
        for j,architecture in enumerate(['cnn','lstm']):
            selected = [next(r for r in metrics if r['architecture']==architecture and r['mode']==mode) for mode in ['input','output','combined']]
            values = np.array([r['test'][metric] for r in selected])
            bounds = np.array([intervals[r['experiment']][metric+'_95_interval'] for r in selected]).T
            axis.bar(np.arange(3)+(j-.5)*.34,values,.34,color=COLORS[j],label=architecture.upper(),yerr=np.vstack([values-bounds[0],bounds[1]-values]),capsize=3)
        axis.set_xticks(range(3),['Prompt','Response','Both'])
        axis.set_ylim(0,1.03)
        axis.set_title(title,loc='left',fontweight='bold')
        axis.grid(axis='y',alpha=.15)
    axes[0].axhline(1/3,color='#777777',linestyle='--',linewidth=1,label='Uniform chance')
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.results/'figures'/'main_results.png',bbox_inches='tight')
    plt.close(fig)

    fig,axes = plt.subplots(2,3,figsize=(9.5,6.0))
    for i,architecture in enumerate(['cnn','lstm']):
        for j,mode in enumerate(['input','output','combined']):
            result = next(r for r in metrics if r['architecture']==architecture and r['mode']==mode)
            cm = np.array(result['confusion_matrix'])
            axis = axes[i,j]
            axis.imshow(cm/cm.sum(axis=1,keepdims=True),cmap='Blues',vmin=0,vmax=1)
            for a in range(3):
                for b in range(3):
                    axis.text(b,a,str(cm[a,b]),ha='center',va='center',color='white' if cm[a,b]/cm[a].sum()>.5 else '#18303c')
            axis.set_xticks(range(3),[n.title() for n in LABELS],fontsize=8)
            axis.set_yticks(range(3),[n.title() for n in LABELS],fontsize=8)
            axis.set_title(f'{architecture.upper()} / {mode}',fontsize=10)
            axis.set_xlabel('Predicted',fontsize=8)
            if j==0:
                axis.set_ylabel('Actual',fontsize=8)
    fig.tight_layout()
    fig.savefig(args.results/'figures'/'confusion_matrices.png',bbox_inches='tight')
    plt.close(fig)

    fig,axes = plt.subplots(2,3,figsize=(9.5,5.6))
    for i,architecture in enumerate(['cnn','lstm']):
        for j,mode in enumerate(['input','output','combined']):
            result = next(r for r in metrics if r['architecture']==architecture and r['mode']==mode)
            history = result['history']
            axis = axes[i,j]
            axis.plot([h['epoch'] for h in history],[h['train_loss'] for h in history],label='Training',color=COLORS[0])
            axis.plot([h['epoch'] for h in history],[h['validation']['loss'] for h in history],label='Validation',color=COLORS[1])
            axis.axvline(result['best_epoch'],color='#999999',linestyle=':',linewidth=1)
            axis.set_title(f'{architecture.upper()} / {mode}',fontsize=10)
            axis.set_xlabel('Epoch',fontsize=8)
            if j==0:
                axis.set_ylabel('Cross-entropy',fontsize=8)
    axes[0,0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.results/'figures'/'training_curves.png',bbox_inches='tight')
    plt.close(fig)

    feature_names = ['log_words','log_characters','type_token_ratio','lines_per_word','bold_markers_per_word','code_fences_per_word','headings_per_word','list_lines_per_word','exclamations_per_word','questions_per_word']
    features = np.array([style_features(text) for text in frame.LLM_output])
    style_model = make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1000,random_state=448))
    style_model.fit(features[train],frame.LLM_name[train])
    style_scores = score(frame.LLM_name[test],style_model.predict(features[test]))
    length_model = make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1000,random_state=448))
    length_model.fit(features[train,:1],frame.LLM_name[train])
    length_scores = score(frame.LLM_name[test],length_model.predict(features[test,:1]))
    description = {}
    for family in LABELS:
        group = frame[frame.LLM_name.eq(family)]
        description[family] = {
            'mean_response_words':float(group.response_word_count.mean()),'median_response_words':float(group.response_word_count.median()),
            'mean_prompt_words':float(group.prompt_word_count.mean()),
            'bold_fraction':float(group.LLM_output.str.contains('**',regex=False).mean()),
            'code_fence_fraction':float(group.LLM_output.str.contains('```',regex=False).mean()),
            'median_type_token_ratio':float(np.median(features[frame.LLM_name.eq(family),2])),
            'is_code_fraction':float(group.is_code.mean()),
        }
    fig,axes = plt.subplots(1,2,figsize=(9.5,3.5))
    axes[0].boxplot([frame.loc[frame.LLM_name.eq(f),'response_word_count'] for f in LABELS],tick_labels=[f.title() for f in LABELS],showfliers=False)
    axes[0].set_title('Response length',loc='left',fontweight='bold')
    axes[0].set_ylabel('Words (whitespace-separated)')
    for offset,key,name,color in [(-.18,'bold_fraction','Bold formatting',COLORS[0]),(.18,'code_fence_fraction','Code fences',COLORS[1])]:
        axes[1].bar(np.arange(3)+offset,[description[f][key]*100 for f in LABELS],width=.35,label=name,color=color)
    axes[1].set_xticks(range(3),[f.title() for f in LABELS])
    axes[1].set_ylim(0,100)
    axes[1].set_ylabel('Responses (%)')
    axes[1].set_title('Formatting frequency',loc='left',fontweight='bold')
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.results/'figures'/'style_analysis.png',bbox_inches='tight')
    plt.close(fig)

    # Frozen-model diagnostic: same retained output window, punctuation/newlines removed.
    ablations = {}
    y = frame.LLM_name.map(dict(zip(LABELS,range(3)))).to_numpy()
    for architecture in ('cnn','lstm'):
        checkpoint = torch.load(args.results/'checkpoints'/f'{architecture}_output.pt',map_location='cpu',weights_only=True)
        model = (TextCNN if architecture=='cnn' else TextLSTM)(len(checkpoint['vocabulary']),checkpoint['embedding_dim'],checkpoint['hidden_dim'],checkpoint['dropout'])
        model.load_state_dict(checkpoint['state_dict'])
        retained = [tokenize(t)[:checkpoint['response_limit']] for t in frame.LLM_output]
        stripped = [[t for t in tokens if t!='<NL>' and re.search(r'\w',t)] for tokens in retained]
        sequences = [encode([],tokens,'output',checkpoint['vocabulary'],checkpoint['prompt_limit'],checkpoint['response_limit']) for tokens in stripped]
        loader = DataLoader(TextDataset(sequences,y,np.flatnonzero(test)),batch_size=64,collate_fn=collate)
        evaluation = evaluate(model,loader)
        ablations[architecture] = {k:evaluation[k] for k in ('accuracy','macro_f1')}

    analysis = {'description':description,'style_features':feature_names,'style_only_classifier':style_scores,'length_only_classifier':length_scores,
                'style_classifier_input':'Full responses; unlike neural models, these auxiliary classifiers see the full response length.',
                'formatting_ablation':ablations,
                'ablation_method':'Remove punctuation and newline tokens from the already-retained 256-token response window; evaluate unchanged output-only models. No retraining or test-set tuning.',
                'ablation_limitation':'This changes token positions, sequence length, and input distribution; it is a robustness diagnostic, not proof that formatting alone causes the original accuracy.'}
    (args.results/'style_analysis.json').write_text(json.dumps(analysis,indent=2)+'\n')
    print(json.dumps({'style_only':style_scores,'length_only':length_scores,'formatting_ablation':ablations}),flush=True)

if __name__=='__main__':
    main()
