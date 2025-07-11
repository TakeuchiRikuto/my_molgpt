from tdc.generation import MolGen

# MOSESデータを取得
data = MolGen(name='MOSES')
split = data.get_split()

# 構造を確認
print("Split keys:", split.keys())
print("\nTrain data columns:", split['train'].columns.tolist())
print("Train data shape:", split['train'].shape)
print("\nFirst few rows:")
print(split['train'].head())

# 他のsplitも確認
for key in split.keys():
    print(f"\n{key} shape:", split[key].shape)
    print(f"{key} columns:", split[key].columns.tolist())