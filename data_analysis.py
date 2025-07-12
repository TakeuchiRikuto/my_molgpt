import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from rdkit import Chem

def analyze_moses2_dataset():
    """moses2.csvの分布を分析して可視化"""
    
    # データ読み込み
    print("Loading moses2.csv...")
    data = pd.read_csv('moses2.csv')
    print(f"Total molecules: {len(data)}")
    print(f"Columns: {data.columns.tolist()}")
    
    # 基本統計
    print("\n=== Basic Statistics ===")
    print(data.describe())
    
    # 分割の確認
    print("\n=== Split Distribution ===")
    print(data['split'].value_counts())
    
    # SMILES長さの計算
    print("\nCalculating SMILES lengths...")
    pattern = r"(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    data['smiles_length'] = data['smiles'].apply(lambda x: len(regex.findall(x.strip())))
    
    # 分子量計算はスキップ（時間がかかりすぎるため）
    print("Skipping molecular weight calculation (too time-consuming for 138万 molecules)")
    # 代わりにSMILES文字列長を複雑さの指標として使用
    
    # 5つのサブプロット作成（分子量を除外）
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('MOSES2 Dataset Distribution Analysis', fontsize=16)
    
    # (a) LogP分布
    axes[0,0].hist(data['logp'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,0].set_title('(a) LogP Distribution')
    axes[0,0].set_xlabel('LogP')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].grid(True, alpha=0.3)
    
    # (b) SMILES文字列長分布（分子の複雑さの指標）
    axes[0,1].hist(data['smiles'].str.len(), bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[0,1].set_title('(b) SMILES String Length Distribution')
    axes[0,1].set_xlabel('SMILES String Length (characters)')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].grid(True, alpha=0.3)
    
    # (c) QED分布
    axes[0,2].hist(data['qed'], bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0,2].set_title('(c) QED Distribution')
    axes[0,2].set_xlabel('QED')
    axes[0,2].set_ylabel('Frequency')
    axes[0,2].grid(True, alpha=0.3)
    
    # (d) SA分布
    axes[1,0].hist(data['sa'], bins=50, alpha=0.7, color='gold', edgecolor='black')
    axes[1,0].set_title('(d) SA Distribution')
    axes[1,0].set_xlabel('Synthetic Accessibility')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].grid(True, alpha=0.3)
    
    # (e) SMILES長さ分布
    axes[1,1].hist(data['smiles_length'], bins=50, alpha=0.7, color='orchid', edgecolor='black')
    axes[1,1].set_title('(e) SMILES Length Distribution')
    axes[1,1].set_xlabel('SMILES Length (tokens)')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].grid(True, alpha=0.3)
    
    # (f) プロパティの相関マトリックス（分子量除外）
    corr_data = data[['qed', 'logp', 'sa', 'smiles_length']].corr()
    im = axes[1,2].imshow(corr_data, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    axes[1,2].set_title('(f) Property Correlations')
    axes[1,2].set_xticks(range(len(corr_data.columns)))
    axes[1,2].set_yticks(range(len(corr_data.columns)))
    axes[1,2].set_xticklabels(corr_data.columns, rotation=45)
    axes[1,2].set_yticklabels(corr_data.columns)
    
    # 相関値をテキストで表示
    for i in range(len(corr_data.columns)):
        for j in range(len(corr_data.columns)):
            axes[1,2].text(j, i, f'{corr_data.iloc[i, j]:.2f}', 
                         ha='center', va='center', fontsize=8)
    
    plt.colorbar(im, ax=axes[1,2], shrink=0.8)
    
    plt.tight_layout()
    plt.savefig('moses2_distribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 統計サマリー出力（分子量除外）
    print("\n=== Property Statistics ===")
    properties = ['qed', 'logp', 'sa', 'smiles_length']
    for prop in properties:
        if prop in data.columns and not data[prop].isna().all():
            print(f"{prop.upper()}:")
            print(f"  Mean: {data[prop].mean():.3f}")
            print(f"  Std:  {data[prop].std():.3f}")
            print(f"  Min:  {data[prop].min():.3f}")
            print(f"  Max:  {data[prop].max():.3f}")
            print()
    
    # 有効分子の確認
    print("=== Validity Check ===")
    valid_count = 0
    total_count = len(data)
    
    for smiles in data['smiles'].head(100):  # サンプルチェック
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            valid_count += 1
    
    print(f"Validity (sample): {valid_count}/100 = {valid_count}%")
    
    return data

if __name__ == "__main__":
    # 実行
    dataset = analyze_moses2_dataset()
    print("Analysis completed!")