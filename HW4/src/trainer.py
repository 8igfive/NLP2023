import pdb
import logging
import math
import random
import torch
import time

from preprocess import tokenize

logger = logging.getLogger(__name__)

class Corpus:
    def __init__(self, seq_len: int, step_interval: int, batch_size: int, tokenize_type: str = 'char', eval_p: float = 0.05):
        self.tokenize_type = tokenize_type
        self.corpus, self.token2id, self.id2token = tokenize(tokenize_type)
        self.corpus['jinyong'], self.eval_corpus = self.corpus['jinyong'][:int(len(self.corpus['jinyong']) * (1 - eval_p))], \
            self.corpus['jinyong'][int(len(self.corpus['jinyong']) * (1 - eval_p)): ]

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.step_interval = step_interval

        logger.info('Token count:')
        logger.info(f"JinYong: {len(self.corpus['jinyong'])}")
        logger.info(f"GuLong: {len(self.corpus['gulong'])}")
        logger.info(f"Vocab size: {len(self.token2id)}")

    def get_train_data(self):
        batch_size_jy = int(self.batch_size * 0.7)
        batch_size_gl = self.batch_size - batch_size_jy

        jy_i = 0
        while jy_i < len(self.corpus['jinyong']) - self.seq_len:
            jy_is = list(range(jy_i, 
                         min(len(self.corpus['jinyong']) - self.seq_len, jy_i + batch_size_jy * self.step_interval),
                         self.step_interval))
            jy_i = jy_is[-1] + self.step_interval
            gl_is = [random.randint(0, len(self.corpus['gulong']) - self.seq_len) for _ in range(batch_size_gl)]

            input_ids = [[self.token2id[token] for token in self.corpus['jinyong'][left_i: left_i + self.seq_len]] 
                         for left_i in jy_is]
            target_ids = [[self.token2id[token] for token in self.corpus['jinyong'][left_i + 1: left_i + 1 + self.seq_len]] 
                          for left_i in jy_is]

            input_ids.extend([[self.token2id[token] for token in self.corpus['gulong'][left_i: left_i + self.seq_len]] 
                              for left_i in gl_is])
            target_ids.extend([[self.token2id[token] for token in self.corpus['gulong'][left_i + 1: left_i + 1 + self.seq_len]] 
                               for left_i in gl_is])
            
            yield torch.tensor(input_ids, dtype=torch.long), torch.tensor(target_ids, dtype=torch.long)

    def __len__(self):
        return math.ceil((len(self.corpus['jinyong']) - self.seq_len) // self.step_interval / int(self.batch_size * 0.7))

    def get_eval_data(self):
        return torch.tensor([self.token2id[token] for token in self.eval_corpus], dtype=torch.long)
    

class Trainer:
    def __init__(self, model, corpus, epoch, lr, wu_steps, device):

        self.device = device

        self.model = model.to(device=device)
        self.corpus = corpus
        self.step_per_epoch = len(self.corpus)
        self.epoch = epoch
        self.min_lr, self.max_lr = lr

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.max_lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda cur_iter: cur_iter / wu_steps if cur_iter < wu_steps else
                (self.min_lr + 0.5*(self.max_lr - self.min_lr) * 
                 (1 + math.cos(((cur_iter - wu_steps) / (self.step_per_epoch * epoch - wu_steps) * math.pi)))) / 
                 self.max_lr
        )
        self.loss_log = []

    def train(self, check_interval=100):
        global_step = 0
        min_loss = float('inf')
        start_time = time.time()
        save_dir = f'../resources/ckpt/{self.corpus.tokenize_type}'

        torch.save({
            'token2id': self.corpus.token2id,
            'id2token': self.corpus.id2token,
            'eval_corpus': self.corpus.eval_corpus, 
            'train_corpus': self.corpus.corpus
        }, f'{save_dir}/other.ckpt')

        for epoch in range(self.epoch):
            for input_ids, target_ids in self.corpus.get_train_data():
                input_ids = input_ids.to(device=self.device)
                target_ids = target_ids.to(device=self.device)

                _, loss = self.model(input_ids, target_ids)

                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()

                self.loss_log.append(loss.item())
                global_step += 1
                if global_step % check_interval == 0:
                    if self.loss_log[-1] < min_loss:
                        suffix = f", model saved to {save_dir}/min_loss.ckpt"
                        torch.save(self.model.state_dict(), f'{save_dir}/min_loss.ckpt')
                        min_loss = self.loss_log[-1]
                    else:
                        suffix = ''
                    step_time = (time.time() - start_time) / global_step
                    logger.info(f'Epoch: {epoch}, GlobalStep: {global_step}, lr: {self.scheduler.get_last_lr()[0]:.5f}, eta: {step_time * (self.step_per_epoch * self.epoch - global_step) / 3600:.3f}h, loss: {self.loss_log[-1]:.4f}{suffix}')
            torch.save(self.model.state_dict(), f'{save_dir}/{epoch}.ckpt')
            logger.info(f'Save Model of Epoch: {epoch} to {save_dir}/{epoch}.ckpt')