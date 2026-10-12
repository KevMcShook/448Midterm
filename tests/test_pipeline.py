"""Small correctness checks, not a replacement for held-out evaluation."""
import sys
from pathlib import Path
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from train_experiments import TextCNN,TextLSTM,build_vocab,encode,seed_everything

class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_vocabulary_never_reads_heldout_examples(self):
        vocab = build_vocab([['train','train'],['heldout','heldout']], [['reply','reply'],['secret','secret']], [0])
        self.assertIn('train',vocab)
        self.assertNotIn('heldout',vocab)
        self.assertNotIn('secret',vocab)

    def test_combined_preserves_response_budget(self):
        vocab = {'p':4,'r':5}
        self.assertEqual(encode(['p']*20,['r']*20,'combined',vocab,2,3),[4,4,2,5,5,5])

    def test_padding_does_not_change_prediction(self):
        for model_type in (TextCNN,TextLSTM):
            seed_everything(448)
            model = model_type(20).eval()
            for length in (1,3,7):
                x = torch.tensor([[4]*length+[0]*max(0,5-length)])
                extended = torch.nn.functional.pad(x,(0,11))
                with torch.no_grad():
                    left = model(x,torch.tensor([length]))
                    right = model(extended,torch.tensor([length]))
                torch.testing.assert_close(left,right,atol=1e-6,rtol=1e-5)

    def test_both_networks_can_learn_simple_distinguishing_signal(self):
        for model_type in (TextCNN,TextLSTM):
            seed_everything(448)
            model = model_type(10,embedding_dim=16,hidden_dim=16,dropout=0)
            x = torch.tensor([[4+i]*8 for i in range(3) for _ in range(4)])
            y = torch.tensor([i for i in range(3) for _ in range(4)])
            lengths = torch.full((len(x),),8)
            optimizer = torch.optim.Adam(model.parameters(),lr=0.03)
            for _ in range(30):
                optimizer.zero_grad()
                loss = torch.nn.functional.cross_entropy(model(x,lengths),y)
                loss.backward()
                optimizer.step()
            self.assertGreater(float((model(x,lengths).argmax(1)==y).float().mean()),0.95)

if __name__ == '__main__':
    unittest.main()
