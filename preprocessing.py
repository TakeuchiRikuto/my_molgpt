#!/usr/bin/env python3
"""
MOSES dataset preprocessing script
Calculates molecular properties and scaffolds, saves to CSV for reuse
"""

import pandas as pd
import numpy as np
from tdc.generation import MolGen
from tdc import Oracle
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import argparse
from tqdm import tqdm
import os

def safe_scaffold(smiles):
    """Safely calculate Murcko scaffold"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        return MurckoScaffold.MurckoScaffoldSmiles(smiles)
    except:
        return ""

def calculate_properties_batch(smiles_list, batch_size=1000):
    """Calculate properties in batches to avoid memory issues"""
    print(f"Calculating properties for {len(smiles_list)} molecules...")
    
    # Oracle初期化
    qed_oracle = Oracle(name='QED')
    logp_oracle = Oracle(name='LogP') 
    sa_oracle = Oracle(name='SA')
    
    all_qed = []
    all_logp = []
    all_sa = []
    
    # バッチ処理
    for i in tqdm(range(0, len(smiles_list), batch_size), desc="Processing batches"):
        batch = smiles_list[i:i+batch_size]
        
        # プロパティ計算
        qed_batch = qed_oracle(batch)
        logp_batch = logp_oracle(batch)
        sa_batch = sa_oracle(batch)
        
        all_qed.extend(qed_batch)
        all_logp.extend(logp_batch)
        all_sa.extend(sa_batch)
    
    return all_qed, all_logp, all_sa

def preprocess_moses(output_file='moses2.csv', debug=False):
    """Main preprocessing function"""
    print("Loading MOSES dataset from TDC...")
    
    # TDCからデータ取得
    data = MolGen(name='MOSES')
    split = data.get_split()
    
    all_dataframes = []
    
    # 各splitを処理
    for split_name in ['train', 'test', 'valid']:
        print(f"\nProcessing {split_name} split...")
        df = split[split_name].copy()
        df['split'] = split_name
        
        # デバッグモード
        if debug:
            sample_size = {'train': 1000, 'test': 200, 'valid': 200}
            df = df.sample(n=min(sample_size[split_name], len(df)), 
                          random_state=42).reset_index(drop=True)
            print(f"Debug mode: using {len(df)} samples")
        
        # プロパティ計算
        smiles_list = df['smiles'].tolist()
        qed_values, logp_values, sa_values = calculate_properties_batch(smiles_list)
        
        df['qed'] = qed_values
        df['logp'] = logp_values
        df['sa'] = sa_values
        
        # スキャフォールド計算
        print(f"Calculating scaffolds for {len(df)} molecules...")
        df['scaffold_smiles'] = df['smiles'].apply(safe_scaffold)
        
        all_dataframes.append(df)
        print(f"Completed {split_name}: {len(df)} molecules")
    
    # 全データを結合
    print("\nCombining all splits...")
    final_data = pd.concat(all_dataframes, ignore_index=True)
    
    # 統計情報
    print(f"\nDataset Statistics:")
    print(f"Total molecules: {len(final_data):,}")
    for split_name in ['train', 'test', 'valid']:
        count = len(final_data[final_data['split'] == split_name])
        print(f"{split_name}: {count:,} molecules")
    
    print(f"\nProperty ranges:")
    print(f"QED: {final_data['qed'].min():.3f} - {final_data['qed'].max():.3f}")
    print(f"LogP: {final_data['logp'].min():.3f} - {final_data['logp'].max():.3f}")
    print(f"SA: {final_data['sa'].min():.3f} - {final_data['sa'].max():.3f}")
    
    # CSV保存
    print(f"\nSaving to {output_file}...")
    final_data.to_csv(output_file, index=False)
    
    file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
    print(f"Saved successfully! File size: {file_size:.1f} MB")
    
    return final_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess MOSES dataset')
    parser.add_argument('--output', type=str, default='moses2.csv',
                       help='Output CSV filename')
    parser.add_argument('--debug', action='store_true', default=False,
                       help='Use small subset for debugging')
    
    args = parser.parse_args()
    
    # 実行
    data = preprocess_moses(args.output, args.debug)
    print("\nPreprocessing completed!")
    print("You can now use the CSV file in your training and generation scripts.")