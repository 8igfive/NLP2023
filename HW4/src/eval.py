import pdb
import math
import torch
import random
from torch.nn import functional as F
from typing import Dict
from model import LSTMLanguageModel
from trainer import Corpus

class Eval:
    def __init__(self, model: LSTMLanguageModel, token2id: Dict[str, int], id2token: Dict[int, str]):
        self.model = model
        self.model.eval()
        self.token2id = token2id
        self.id2token = id2token
        self.device = torch.device('cuda:0')

    def generate(self, mode: str, prefix: str, decode_window: int = 128, generate_len: int = 256, **kwargs):
        if mode == 'greedy':
            g_fn = self._greedy
        elif mode == 'beam_search':
            g_fn = self._beam_search
        elif mode == 'sample':
            g_fn = self._sample

        return g_fn(prefix, decode_window, generate_len, **kwargs)
    
    def _greedy(self, prefix: str, decode_window: int = 128, generate_len: int = 256):
        ids_full = torch.tensor([self.token2id[token] for token in prefix], dtype=torch.long, device=self.device)
        log_ps = None
        for _ in range(generate_len):
            input_ids = ids_full[-decode_window: ]
            with torch.no_grad():
                logits, _ = self.model(input_ids)
            ids_full = torch.cat([ids_full, torch.argmax(logits[-1], keepdim=True)], dim=0)
            if log_ps is None:
                log_sm = F.log_softmax(logits, dim=-1) # shape of (seq_len, vocab_size)
                indices = torch.arange(log_sm.shape[0] * log_sm.shape[1]).view_as(log_sm).to(self.device)
                mask = indices % indices.shape[1] == ids_full[1: ][:, None]
                log_ps = log_sm[mask].sum()
            else:
                log_ps += F.log_softmax(logits[-1], dim=-1)[ids_full[-1]]

        return [self.id2token[idx.item()] for idx in ids_full], log_ps.item()

    def _beam_search(self, prefix: str, decode_window: int = 128, generate_len: int = 256, tau: float = 1., k: int = 3):
        ids_full = torch.tensor([self.token2id[token] for token in prefix], 
                                dtype=torch.long, device=self.device).repeat(k, 1) # shape of (k, seq_len)
        log_ps = None
        for i in range(generate_len):
            input_ids = ids_full[:, -decode_window: ]
            with torch.no_grad():
                logits, _ = self.model(input_ids) # shape of (k, decode_windows, vocab_size)
            logits /= tau
            log_sm = F.log_softmax(logits, dim=-1)
            values, indices = log_sm[:, -1].topk(k, dim=-1) # shape of (k, k)
            ids_full = ids_full[:, None].repeat(1, k, 1).reshape(-1, ids_full.shape[-1])
            ids_full = torch.cat([ids_full, indices.reshape(-1)[:, None]], dim=-1)
            if log_ps is None:
                ids_full = ids_full[:k]
                indices = torch.arange(log_sm.shape[0] * log_sm.shape[1] * log_sm.shape[2]).reshape_as(log_sm).to(self.device)
                mask = indices % indices.shape[-1] == ids_full[:, 1: ].unsqueeze(-1)
                log_ps = log_sm[mask].reshape(k, -1).sum(dim=-1) # shape of (k, )
            else:
                log_ps = log_ps[:, None].repeat(1, k).reshape(-1) # shape of (k*k, )
                log_ps += values.reshape(-1)
                log_ps, ids_full_indices = log_ps.topk(k, dim=0)
                ids_full = ids_full.index_select(dim=0, index=ids_full_indices)
            
        final_ids = ids_full[0]
        final_log_p = log_ps[0]
        return [self.id2token[idx.item()] for idx in final_ids], final_log_p.item()

    def _sample(self, prefix: str, decode_window: int = 128, generate_len: int = 256, tau: float = 1.):
        ids_full = torch.tensor([self.token2id[token] for token in prefix], dtype=torch.long, device=self.device)
        log_ps = None
        for _ in range(generate_len):
            input_ids = ids_full[-decode_window: ]
            with torch.no_grad():
                logits, _ = self.model(input_ids)
            logits /= tau
            sm = F.softmax(logits, dim=-1)
            next_token = random.choices(range(sm.shape[-1]), weights=sm[-1], k=1)[0]
            next_p = sm[-1][next_token]
            ids_full = torch.cat([ids_full, torch.tensor([next_token], dtype=torch.long, device=self.device)], dim=0)
            if log_ps is None:
                log_sm = torch.log(sm)
                indices = torch.arange(log_sm.shape[0] * log_sm.shape[1]).reshape_as(log_sm).to(self.device)
                mask = indices % indices.shape[-1] == ids_full[1: ][:, None]
                log_ps = log_sm[mask].sum()
            else:
                log_ps += torch.log(next_p.to(log_ps))
        
        return [self.id2token[idx.item()] for idx in ids_full], log_ps.item()
                        

    def perplexity(self, text: str, window_size: int =128, tau: float = 1.):
        ids_full = torch.tensor([self.token2id[token] for token in text], dtype=torch.long, device=self.device)
        log_ps = None
        for i in range(ids_full.shape[0] - window_size):
            input_ids = ids_full[i: i+window_size]
            with torch.no_grad():
                logits, _ = self.model(input_ids) # shape of (k, decode_windows, vocab_size)
            logits /= tau
            log_sm = F.log_softmax(logits, dim=-1)
            if log_ps is None:
                indices = torch.arange(log_sm.shape[0] * log_sm.shape[1]).reshape_as(log_sm).to(self.device)
                mask = indices % indices.shape[-1] == ids_full[i + 1: i + 1 + window_size][:, None]
                log_ps = log_sm[mask].sum()
            else:
                log_ps += log_sm[-1][ids_full[i + window_size]]

        return -1 / len(text) * log_ps.item()

if __name__ == '__main__':
    DATA_DIR = r'../resources'
    CKPT_DIR = r'ckpt/char/384_12'
    ckpt_path = rf'{DATA_DIR}/{CKPT_DIR}/min_loss.ckpt'
    model = LSTMLanguageModel(5772, 384, 12).cuda()
    model.load_state_dict(torch.load(ckpt_path))

    other = torch.load(fr'{DATA_DIR}/{CKPT_DIR}/other.ckpt')

    eval = Eval(model, other['token2id'], other['id2token'])

    # prefix = ''.join(other['train_corpus']['jinyong'][449: 549])
    # prefix = ''.join(other['eval_corpus'][3:103])
    prefix = ''.join(other['eval_corpus'][3:1003])

    print(f'Prefix:\n{prefix}')

    # output, log_p = eval.generate('greedy', prefix)
    # print(f"Greedy Output:\n{''.join(output)}\nLog_p: {log_p}")

    # output, log_p = eval.generate('beam_search', prefix, k=5)
    # print(f"Beam Output:\n{''.join(output)}\nLog_p: {log_p}")

    # output, log_p = eval.generate('sample', prefix,)
    # print(f"Sample Output:\n{''.join(output)}\nLog_p: {log_p}")

    # for i in range(6):
    #     print(f'================CHECKPOINT {i}================')
    #     eval.model.load_state_dict(torch.load(rf'{DATA_DIR}/{CKPT_DIR}/{i}.ckpt'))
        
    #     output, log_p = eval.generate('greedy', prefix)
    #     print(f"Greedy Output:\n{''.join(output)}\nLog_p: {log_p}")

    #     output, log_p = eval.generate('beam_search', prefix, k=5)
    #     print(f"Beam Output:\n{''.join(output)}\nLog_p: {log_p}")

    #     output, log_p = eval.generate('sample', prefix,)
    #     print(f"Sample Output:\n{''.join(output)}\nLog_p: {log_p}")

    # for i in range(-2, 3):
    #     print(f"================tau {10**i}================")
    #     output, log_p = eval.generate('sample', prefix, tau=10**i)
    #     print(f"Sample Output:\n{''.join(output)}\nLog_p: {log_p}")

    for i in range(6):
        eval.model.load_state_dict(torch.load(rf'{DATA_DIR}/{CKPT_DIR}/{i}.ckpt'))
        perplexity = eval.perplexity(prefix, 256)
        print(f'{i}\t\t{perplexity}')

    pdb.set_trace()
