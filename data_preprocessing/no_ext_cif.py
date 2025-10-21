import pandas as pd

df = pd.read_csv('id_prop.csv', header=None)
df[0] = df[0].str.replace(r'\.cif$', '', regex=True)
df.to_csv('id_prop.csv', header=False, index=False)