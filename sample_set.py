from tdc import Oracle

# 基本的なオラクル
qed = Oracle('QED')  # 薬物らしさ
logp = Oracle('LogP')  # 脂溶性
sa = Oracle('SA')  # 合成容易性

# テスト分子で確認
test_smiles = ['CCO', 'c1ccccc1', 'CC(=O)O']
print("QED:", qed(test_smiles))
print("LogP:", logp(test_smiles))
print("SA:", sa(test_smiles))