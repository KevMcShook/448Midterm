"""Verify the saved split and recompute metrics from saved predictions."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path('data/cleaned/claude_gemini_qwen_balanced.parquet'))
    parser.add_argument('--results',type=Path,default=Path('results'))
    args = parser.parse_args()
    frame = pd.read_parquet(args.data).set_index('example_id')
    config = json.loads((args.results/'config.json').read_text())
    assert hashlib.sha256(args.data.read_bytes()).hexdigest() == config['data_sha256']
    split = pd.read_parquet(args.results/'split_manifest.parquet').set_index('example_id')
    assert split.index.is_unique and set(split.index)==set(frame.index)
    assert split.LLM_name.eq(frame.loc[split.index].LLM_name).all()
    full = frame.join(split['split'])
    for col in ['split_group','prompt_id','response_id','evaluation_session_id']:
        assert full.groupby(col)['split'].nunique().max()==1, f'Cross-partition overlap: {col}'
    test_ids = set(split.index[split['split'].eq('test')])
    labels = ['claude','gemini','qwen']
    results = json.loads((args.results/'all_metrics.json').read_text())
    assert {r['experiment'] for r in results} == {a+'_'+m for a in ['cnn','lstm'] for m in ['input','output','combined']}
    for result in results:
        predictions = pd.read_json(args.results/'predictions'/f"{result['experiment']}.json").set_index('example_id')
        assert predictions.index.is_unique and set(predictions.index)==test_ids
        assert predictions.true_label.eq(frame.loc[predictions.index].LLM_name).all()
        assert predictions.split_group.eq(frame.loc[predictions.index].split_group).all()
        prob = predictions[[f'probability_{label}' for label in labels]].to_numpy()
        assert np.allclose(prob.sum(axis=1),1,atol=1e-6)
        assert (np.asarray(labels)[prob.argmax(axis=1)]==predictions.predicted_label.to_numpy()).all()
        truth,pred = predictions.true_label,predictions.predicted_label
        assert abs(accuracy_score(truth,pred)-result['test']['accuracy'])<1e-12
        assert abs(f1_score(truth,pred,labels=labels,average='macro')-result['test']['macro_f1'])<1e-12
        assert confusion_matrix(truth,pred,labels=labels).tolist()==result['confusion_matrix']
        selected = next(h for h in result['history'] if h['epoch']==result['best_epoch'])
        assert abs(selected['validation']['macro_f1']-max(h['validation']['macro_f1'] for h in result['history']))<1e-5
    verification = {'data_hash_matches':True,'six_experiments_complete':True,'no_group_prompt_response_session_overlap':True,'prediction_ids_and_true_labels_match':True,'probabilities_and_predictions_agree':True,'metrics_recomputed_successfully':True,'validation_checkpoint_selection_verified':True,'test_examples':len(test_ids)}
    (args.results/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    print(json.dumps(verification,indent=2))

if __name__=='__main__':
    main()
