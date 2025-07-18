"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import math
import logging

from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
from torch.cuda.amp import GradScaler

from utils import check_novelty, sample, canonic_smiles
import re
import pandas as pd
from rdkit import Chem


logger = logging.getLogger(__name__)

class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0 # for DataLoader

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config, stoi, itos):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config

        # take over whatever gpus are on the system
        self.device = 'cpu'
        self.stoi = stoi
        self.itos = itos

        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            # self.model = torch.nn.DataParallel(self.model).to(self.device)
            self.model = self.model.to(self.device)

    def save_checkpoint(self, epoch, train_loss):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info("saving %s", self.config.ckpt_path)
        torch.save({
            'model_state_dict': raw_model.state_dict(),
            'model_config': self.model.config.__dict__,  # 設定全体を保存
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
            'train_loss': train_loss
        }, self.config.ckpt_path)

    def train(self, wandb):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        self.optimizer = raw_model.configure_optimizers(config)
        scaler = GradScaler()

        # [' ', '#', '(', ')', '-', '1', '2', '3', '4', '5', '6', '<', '=', 'B', 'C', 'F', 'H', 'N', 'O', 'S', '[', ']', 'c', 'l', 'n', 'o', 'r', 's']
        # ['#', '(', ')', '-', '1', '2', '3', '4', '5', '6', '<', '=', 'Br', 'C', 'Cl', 'F', 'N', 'O', 'S', '[H]', '[nH]', 'c', 'n', 'o', 's']


        def run_epoch(split):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)

            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            for it, (x, y, p, scaffold) in pbar:

                # place data on the correct device
                x = x.to(self.device)
                y = y.to(self.device)
                p = p.to(self.device)
                scaffold = scaffold.to(self.device) 

                # forward the model
                with torch.cuda.amp.autocast():
                    with torch.set_grad_enabled(is_train):
                        logits, loss, _ = model(x, y, p, scaffold)
                        loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                        losses.append(loss.item())

                if is_train:

                    # backprop and update the parameters
                    model.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    scaler.step(self.optimizer)
                    scaler.update()

                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (y >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    # report progress
                    wandb.log({'step_train_loss': loss, 'train_step': it + epoch*len(loader), 'learning_rate': lr})
                    pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")
    
            if is_train:
                return float(np.mean(losses))

            if not is_train:
                test_loss = float(np.mean(losses))
                logger.info("test loss: %f", test_loss)
                return test_loss

        best_loss = float('inf')
        self.tokens = 0 # counter used for learning rate decay
        molecules = []
        training_logs = []#追加

        for epoch in range(config.max_epochs):

            train_loss = run_epoch('train')
            if self.test_dataset is not None:
                test_loss = run_epoch('test')
                
            # ログを記録 ← これを追加
            training_logs.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'test_loss': test_loss if self.test_dataset is not None else None
            })

            wandb.log({'epoch_valid_loss': test_loss, 'epoch_train_loss': train_loss, 'epoch': epoch + 1})

            # supports early stopping based on the test loss, or just save always if no test set is provided
            good_model = self.test_dataset is None or test_loss < best_loss
            if self.config.ckpt_path is not None and good_model:
                best_loss = test_loss
                print(f'Saving at epoch {epoch + 1}')
                self.save_checkpoint(epoch + 1, train_loss)

            if self.config.generate:
                pattern =  r"(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
                regex = re.compile(pattern)
                context = "C"
                for i in range(2):
                    x = torch.tensor([self.stoi[s] for s in regex.findall(context)], dtype=torch.long)[None,...].repeat(512, 1).to('cuda')
                    p = None
                    sca = None
                    y = sample(model, x, self.config.block_size, temperature=0.8, sample=True, top_k=10, prop = p, scaffold = sca)
                    for gen_mol in y:
                        completion = ''.join([self.itos[int(i)] for i in gen_mol])
                        completion = completion.replace('<', '')
                        mol = get_mol(completion)
                        if mol:
                            smiles = Chem.MolToSmiles(mol)
                            molecules.append((mol, smiles, epoch))

        if self.config.generate:
            df = pd.DataFrame(molecules, columns = ['molecule', 'smiles', 'epoch'])
            return df

        return pd.DataFrame(training_logs)
