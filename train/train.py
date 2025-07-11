import pandas as pd
import argparse
from utils import set_seed
import numpy as np
import wandb
import os
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.cuda.amp import GradScaler

from model import GPT, GPTConfig
from trainer import Trainer, TrainerConfig
from dataset import SmileDataset
import math
from utils import SmilesEnumerator
import re

from tdc.generation import MolGen
from tdc import Oracle
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def create_vocab_from_data(smiles_list, scaffold_list=None):
        all_text = ' '.join(smiles_list)
        if scaffold_list:
            all_text += ' '.join(scaffold_list)
    
        tokens = regex.findall(all_text)
        vocab = sorted(list(set(tokens)))
        return vocab

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--run_name', type=str,
                        help="name for wandb run", required=False)
    parser.add_argument('--debug', action='store_true',
                        default=False, help='debug')
    # in moses dataset, on average, there are only 5 molecules per scaffold
    parser.add_argument('--scaffold', action='store_true',
                        default=False, help='condition on scaffold')
    parser.add_argument('--lstm', action='store_true',
                        default=False, help='use lstm for transforming scaffold')
    parser.add_argument('--data_name', type=str, default='moses2',
                        help="name of the dataset to train on", required=False)
    # parser.add_argument('--property', type=str, default = 'qed', help="which property to use for condition", required=False)
    parser.add_argument('--props', nargs="+", default=['qed'],
                        help="properties to be used for condition", required=False)
    parser.add_argument('--num_props', type=int, default = 0, help="number of properties to use for condition", required=False)
    # parser.add_argument('--prop1_unique', type=int, default = 0, help="unique values in that property", required=False)
    parser.add_argument('--n_layer', type=int, default=8,
                        help="number of layers", required=False)
    parser.add_argument('--n_head', type=int, default=8,
                        help="number of heads", required=False)
    parser.add_argument('--n_embd', type=int, default=256,
                        help="embedding dimension", required=False)
    parser.add_argument('--max_epochs', type=int, default=10,
                        help="total epochs", required=False)
    parser.add_argument('--batch_size', type=int, default=512,
                        help="batch size", required=False)
    parser.add_argument('--learning_rate', type=float,
                        default=6e-4, help="learning rate", required=False)
    parser.add_argument('--lstm_layers', type=int, default=0,
                        help="number of layers in lstm", required=False)

    args = parser.parse_args()

    set_seed(42)

    wandb.init(project="lig_gpt", name=args.run_name)


##########################################################    # Load data

    # TDCを使用した場合
    # MOSESデータセット
    if 'moses' in args.data_name:
        data = pd.read_csv(f'{args.data_name}.csv')
        train_data = data[data['split'] == 'train'].copy()
        val_data = data[data['split'] == 'test'].copy()
    
        # デバッグモード対応
        if args.debug:
            print("Debug mode: using subset of data")
            train_data = train_data.sample(n=1000, random_state=42).reset_index(drop=True)
            val_data = val_data.sample(n=200, random_state=42).reset_index(drop=True)
        
        print(f"Loaded train: {len(train_data)}, val: {len(val_data)}")
        print("Properties already calculated:", train_data.columns.tolist())
    
    else:
        pass

    # プロパティとスキャフォールドの処理
    if args.num_props > 0:
        # 指定されたプロパティを取得
        available_props = [prop for prop in args.props if prop in train_data.columns]
        if len(available_props) == 0:
            print(f"Warning: None of the requested properties {args.props} found in data. Using default QED.")
            available_props = ['qed']
    
        prop = train_data[available_props].values.tolist()
        vprop = val_data[available_props].values.tolist()
        num_props = len(available_props)
        print(f"Using {num_props} properties: {available_props}")
    else:
        # プロパティを使わない場合
        prop = [[0.0] for _ in range(len(train_data))]  # [[0.0], [0.0], ...]
        vprop = [[0.0] for _ in range(len(val_data))]
        num_props = 0
        print("Training without properties (unconditional generation)")

    # スキャフォールドの処理
    if args.scaffold:
        scaffold = train_data['scaffold_smiles'].tolist()
        vscaffold = val_data['scaffold_smiles'].tolist()
        print("Using scaffold conditioning")
    else:
        scaffold = [""] * len(train_data)
        vscaffold = [""] * len(val_data)
        print("Training without scaffold conditioning")

    # SMILES文字列の取得
    smiles = train_data['smiles'].tolist()
    vsmiles = val_data['smiles'].tolist()

    # 分子の正規表現パターン（SMILES tokenization用）
    pattern = r"(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)

    # 分子の最大長を計算
    print("Calculating molecule lengths...")
    lens = [len(regex.findall(i.strip())) for i in (smiles + vsmiles)]
    max_len = max(lens)
    print('Max molecule length: ', max_len)

    # スキャフォールドの最大長を計算
    if args.scaffold:
        scaffold_lens = [len(regex.findall(i.strip())) for i in (scaffold + vscaffold) if i.strip()]
        scaffold_max_len = max(scaffold_lens) if scaffold_lens else 0
        print('Max scaffold length: ', scaffold_max_len)
    else:
        scaffold_max_len = 0

    # パディング（'<'で埋める）
    print("Padding molecules...")
    smiles = [i + '<' * (max_len - len(regex.findall(i.strip()))) for i in smiles]
    vsmiles = [i + '<' * (max_len - len(regex.findall(i.strip()))) for i in vsmiles]

    if args.scaffold:
        scaffold = [i + '<' * (scaffold_max_len - len(regex.findall(i.strip()))) for i in scaffold]
        vscaffold = [i + '<' * (scaffold_max_len - len(regex.findall(i.strip()))) for i in vscaffold]

    vocab_file = f'{args.data_name}_stoi.json'

    # 既存ファイル削除
    if os.path.exists(vocab_file):
       os.remove(vocab_file)

    whole_string = create_vocab_from_data(smiles, scaffold if args.scaffold else None)

    # ファイル保存
    with open(vocab_file, 'w') as f:
        json.dump(whole_string, f)
    print(f"Created and saved vocabulary to {vocab_file}")

    
    # データセット作成
    train_dataset = SmileDataset(args, smiles, whole_string, max_len, prop=prop, aug_prob=0, scaffold=scaffold, scaffold_maxlen=scaffold_max_len)
    with open(f'{args.data_name}_stoi.json', 'w') as f:
        json.dump(train_dataset.stoi, f)
        print(f"Saved vocabulary to {args.data_name}_stoi.json")
    valid_dataset = SmileDataset(args, vsmiles, whole_string, max_len, prop=vprop, aug_prob=0, scaffold=vscaffold, scaffold_maxlen=scaffold_max_len)

    # モデル設定
    mconf = GPTConfig(train_dataset.vocab_size, train_dataset.max_len, num_props=num_props,
                        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, 
                        scaffold=args.scaffold, scaffold_maxlen=scaffold_max_len,
                        lstm=args.lstm, lstm_layers=args.lstm_layers)
    model = GPT(mconf)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 訓練設定
    tconf = TrainerConfig(max_epochs=args.max_epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
                            lr_decay=True, warmup_tokens=0.1*len(train_data)*max_len, final_tokens=args.max_epochs*len(train_data)*max_len,
                            num_workers=10, ckpt_path=f'./weights/{args.run_name}.pt', block_size=train_dataset.max_len, generate=False)

    # 訓練実行
    trainer = Trainer(model, train_dataset, valid_dataset,
                        tconf, train_dataset.stoi, train_dataset.itos)

    print("Starting training...")
    df = trainer.train(wandb if 'wandb' in locals() else None)

    # 結果保存
    df.to_csv(f'{args.run_name}.csv', index=False)
    print(f"Training completed! Results saved to {args.run_name}.csv")



    