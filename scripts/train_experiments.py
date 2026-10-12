"""Train and evaluate the six required LLM-attribution experiments.

Run from the repository root: python scripts/train_experiments.py
The same saved grouped split and training-only vocabulary serve every model.
"""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import random
import re
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

LABELS = ['claude', 'gemini', 'qwen']
TOKEN = re.compile(r'\w+|[^\w\s]', re.UNICODE)
SPECIAL = {'<PAD>': 0, '<UNK>': 1, '<SEP>': 2, '<NL>': 3}

def tokenize(text):
    tokens = []
    for i, line in enumerate(text.split('\n')):
        if i:
            tokens.append('<NL>')
        tokens.extend(TOKEN.findall(line.lower()))
    return tokens

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

def make_splits(frame, seed=448):
    y = frame.LLM_name.map(dict(zip(LABELS, range(3)))).to_numpy()
    folds = np.zeros(len(frame), dtype=int)
    splitter = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=seed)
    for fold, (_, heldout) in enumerate(splitter.split(np.zeros(len(frame)), y, frame.split_group)):
        folds[heldout] = fold
    split = np.where(folds < 3, 'test', np.where(folds < 6, 'validation', 'train'))
    for key in ('split_group', 'prompt_id', 'response_id', 'evaluation_session_id'):
        check = pd.DataFrame({'key':frame[key], 'split':split}).dropna()
        assert check.groupby('key')['split'].nunique().max() == 1, f'Leakage in {key}'
    assert all(set(y[split == part]) == {0,1,2} for part in ('train','validation','test'))
    return split, y

def build_vocab(prompts, responses, indices, max_vocab=20000):
    counts = Counter()
    for i in indices:
        counts.update(prompts[i])
        counts.update(responses[i])
    vocabulary = dict(SPECIAL)
    for token, count in sorted(counts.items(), key=lambda item: (-item[1],item[0])):
        if count >= 2 and token not in vocabulary:
            vocabulary[token] = len(vocabulary)
            if len(vocabulary) >= max_vocab:
                break
    return vocabulary

def encode(prompt, response, mode, vocab, prompt_limit=128, response_limit=256):
    p = [vocab.get(t,1) for t in prompt[:prompt_limit]]
    r = [vocab.get(t,1) for t in response[:response_limit]]
    values = p if mode == 'input' else r if mode == 'output' else p+[2]+r
    return values or [1]

class TextDataset(Dataset):
    def __init__(self, sequences, labels, indices):
        self.sequences, self.labels, self.indices = sequences, labels, np.asarray(indices)
    def __len__(self):
        return len(self.indices)
    def __getitem__(self,index):
        i = self.indices[index]
        return self.sequences[i], int(self.labels[i]), int(i)

def collate(batch):
    lengths = torch.tensor([len(x[0]) for x in batch], dtype=torch.long)
    tokens = torch.zeros(len(batch), max(5,int(lengths.max())), dtype=torch.long)
    for i,(sequence,_,_) in enumerate(batch):
        tokens[i,:len(sequence)] = torch.tensor(sequence, dtype=torch.long)
    return tokens, lengths, torch.tensor([x[1] for x in batch]), [x[2] for x in batch]

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,embedding_dim,padding_idx=0)
        self.convolutions = nn.ModuleList([nn.Conv1d(embedding_dim,hidden_dim,k) for k in (3,4,5)])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim*3,3)
    def forward(self,tokens,lengths):
        embedded = self.embedding(tokens).transpose(1,2)
        pooled = []
        for convolution in self.convolutions:
            activation = torch.relu(convolution(embedded))
            valid = torch.clamp(lengths-convolution.kernel_size[0]+1,min=1)
            mask = torch.arange(activation.size(2),device=tokens.device)[None,:] >= valid[:,None]
            activation = activation.masked_fill(mask[:,None,:], float('-inf'))
            pooled.append(activation.amax(dim=2))
        return self.classifier(self.dropout(torch.cat(pooled,dim=1)))

class TextLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,embedding_dim,padding_idx=0)
        self.recurrent = nn.LSTM(embedding_dim,hidden_dim,batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim,3)
    def forward(self,tokens,lengths):
        states,_ = self.recurrent(self.embedding(tokens))
        # A forward-only LSTM state at the last real token is unaffected by right padding.
        final = states[torch.arange(tokens.size(0),device=tokens.device),lengths-1]
        return self.classifier(self.dropout(final))

def evaluate(model,loader):
    model.eval()
    truth,predicted,probabilities,indices = [],[],[],[]
    loss_sum = 0.0
    with torch.no_grad():
        for tokens,lengths,y,index in loader:
            logits = model(tokens,lengths)
            loss_sum += nn.functional.cross_entropy(logits,y,reduction='sum').item()
            truth.extend(y.tolist())
            predicted.extend(logits.argmax(dim=1).tolist())
            probabilities.extend(logits.softmax(dim=1).tolist())
            indices.extend(index)
    return {'loss':loss_sum/len(truth),'accuracy':accuracy_score(truth,predicted),
            'macro_f1':f1_score(truth,predicted,labels=[0,1,2],average='macro',zero_division=0),
            'truth':truth,'predicted':predicted,'probabilities':probabilities,'indices':indices}

def metric_summary(evaluation):
    return {k:evaluation[k] for k in ('loss','accuracy','macro_f1')}

def train_one(architecture, mode, sequences, y, indices, vocab, frame, output, args):
    name = f'{architecture}_{mode}'
    result_path = output/'metrics'/f'{name}.json'
    if args.resume and result_path.exists():
        print(json.dumps({'event':'already_complete','experiment':name}),flush=True)
        return json.loads(result_path.read_text())
    seed_everything(args.seed)
    model = (TextCNN if architecture == 'cnn' else TextLSTM)(len(vocab),args.embedding_dim,args.hidden_dim,args.dropout)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(TextDataset(sequences,y,indices['train']),batch_size=args.batch_size,shuffle=True,generator=generator,collate_fn=collate)
    val_loader = DataLoader(TextDataset(sequences,y,indices['validation']),batch_size=args.batch_size,collate_fn=collate)
    test_loader = DataLoader(TextDataset(sequences,y,indices['test']),batch_size=args.batch_size,collate_fn=collate)
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=0.01)
    best_f1,best_epoch,best_state,stale = -1.0,0,None,0
    history = []
    started = time.monotonic()
    for epoch in range(1,args.epochs+1):
        model.train()
        train_loss,n = 0.0,0
        for tokens,lengths,labels,_ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(tokens,lengths)
            loss = nn.functional.cross_entropy(logits,labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            train_loss += loss.item()*len(labels)
            n += len(labels)
        val = evaluate(model,val_loader)
        entry = {'epoch':epoch,'train_loss':train_loss/n,'validation':metric_summary(val),'elapsed_seconds':time.monotonic()-started}
        history.append(entry)
        print(json.dumps({'event':'epoch','experiment':name,**entry}),flush=True)
        if val['macro_f1'] > best_f1 + 1e-5:
            best_f1,best_epoch = val['macro_f1'],epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    model.load_state_dict(best_state)
    validation = evaluate(model,val_loader)
    test = evaluate(model,test_loader)
    result = {'experiment':name,'architecture':architecture,'mode':mode,'seed':args.seed,
              'best_epoch':best_epoch,'epochs_run':len(history),'train_seconds':time.monotonic()-started,
              'parameter_count':sum(p.numel() for p in model.parameters()),
              'validation':metric_summary(validation),'test':metric_summary(test),'history':history,
              'confusion_matrix':confusion_matrix(test['truth'],test['predicted'],labels=[0,1,2]).tolist(),
              'classification_report':classification_report(test['truth'],test['predicted'],labels=[0,1,2],target_names=LABELS,output_dict=True,zero_division=0)}
    result_path.write_text(json.dumps(result,indent=2)+'\n')
    predictions = []
    for i,truth,pred,prob in zip(test['indices'],test['truth'],test['predicted'],test['probabilities']):
        predictions.append({'example_id':frame.iloc[i].example_id,'split_group':frame.iloc[i].split_group,
                            'true_label':LABELS[truth],'predicted_label':LABELS[pred],
                            **{f'probability_{label}':p for label,p in zip(LABELS,prob)}})
    (output/'predictions'/f'{name}.json').write_text(json.dumps(predictions,indent=2)+'\n')
    torch.save({'state_dict':best_state,'architecture':architecture,'mode':mode,'vocabulary':vocab,
                'labels':LABELS,'embedding_dim':args.embedding_dim,'hidden_dim':args.hidden_dim,
                'dropout':args.dropout,'prompt_limit':args.prompt_limit,'response_limit':args.response_limit}, output/'checkpoints'/f'{name}.pt')
    print(json.dumps({'event':'complete','experiment':name,'test':result['test']}),flush=True)
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path('data/cleaned/claude_gemini_qwen_balanced.parquet'))
    parser.add_argument('--output',type=Path,default=Path('results'))
    parser.add_argument('--seed',type=int,default=448)
    parser.add_argument('--epochs',type=int,default=12)
    parser.add_argument('--patience',type=int,default=3)
    parser.add_argument('--batch-size',type=int,default=64)
    parser.add_argument('--embedding-dim',type=int,default=64)
    parser.add_argument('--hidden-dim',type=int,default=64)
    parser.add_argument('--dropout',type=float,default=0.3)
    parser.add_argument('--learning-rate',type=float,default=0.001)
    parser.add_argument('--prompt-limit',type=int,default=128)
    parser.add_argument('--response-limit',type=int,default=256)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    frame = pd.read_parquet(args.data).reset_index(drop=True)
    assert frame.example_id.is_unique and set(frame.LLM_name) == set(LABELS)
    data_hash = hashlib.sha256(args.data.read_bytes()).hexdigest()
    config = {k:(str(v) if isinstance(v,Path) else v) for k,v in vars(args).items() if k not in ('resume','output')}
    config['data'] = args.data.name
    config['data_sha256'] = data_hash
    config['versions'] = {'torch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__,'scikit_learn':sklearn.__version__}
    if (args.output/'config.json').exists():
        previous = json.loads((args.output/'config.json').read_text())
        if not args.resume or previous != config:
            raise RuntimeError('Output already exists or configuration changed. Use a new --output directory or matching --resume.')
    for directory in ('metrics','predictions','checkpoints','figures'):
        (args.output/directory).mkdir(parents=True,exist_ok=True)
    (args.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    split,y = make_splits(frame,args.seed)
    indices = {part:np.flatnonzero(split==part) for part in ('train','validation','test')}
    manifest = frame[['example_id','LLM_name','split_group']].copy()
    manifest['split'] = split
    manifest.to_parquet(args.output/'split_manifest.parquet',index=False)
    summary = {part:{'total':int(len(idx)),'class_counts':dict(Counter(frame.iloc[idx].LLM_name)),'groups':int(frame.iloc[idx].split_group.nunique())} for part,idx in indices.items()}
    (args.output/'split_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    prompts = [tokenize(t) for t in frame.LLM_Input]
    responses = [tokenize(t) for t in frame.LLM_output]
    vocab = build_vocab(prompts,responses,indices['train'])
    (args.output/'vocabulary.json').write_text(json.dumps(vocab,ensure_ascii=False)+'\n')
    truncation = {part:{'prompt_fraction_truncated':float(np.mean([len(prompts[i])>args.prompt_limit for i in idx])),
                       'response_fraction_truncated':float(np.mean([len(responses[i])>args.response_limit for i in idx]))} for part,idx in indices.items()}
    (args.output/'preprocessing.json').write_text(json.dumps({'vocabulary_size':len(vocab),'truncation':truncation},indent=2)+'\n')
    majority = Counter(y[indices['train']]).most_common(1)[0][0]
    majority_prediction = [majority]*len(indices['test'])
    baseline = {'training_majority_label':LABELS[majority],
                'test_accuracy':accuracy_score(y[indices['test']],majority_prediction),
                'test_macro_f1':f1_score(y[indices['test']],majority_prediction,labels=[0,1,2],average='macro',zero_division=0),
                'uniform_random_expected_accuracy':1/3}
    (args.output/'baseline.json').write_text(json.dumps(baseline,indent=2)+'\n')
    print(json.dumps({'event':'setup','split':summary,'vocabulary':len(vocab)}),flush=True)
    results = []
    # Complete output-only models first, then the two prompt comparisons.
    for mode in ('output','input','combined'):
        sequences = [encode(p,r,mode,vocab,args.prompt_limit,args.response_limit) for p,r in zip(prompts,responses)]
        for architecture in ('cnn','lstm'):
            results.append(train_one(architecture,mode,sequences,y,indices,vocab,frame,args.output,args))
    (args.output/'all_metrics.json').write_text(json.dumps(results,indent=2)+'\n')

if __name__ == '__main__':
    main()
