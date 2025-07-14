import pandas as pd
data = pd.read_csv('moses2.csv')
data = data.rename(columns={'scaffold_smiles': 'scaffold'})
data.to_csv('moses2.csv', index=False)
