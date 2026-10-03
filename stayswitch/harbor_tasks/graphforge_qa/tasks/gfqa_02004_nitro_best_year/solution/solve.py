import os,json,pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
d=pd.read_csv(D+'/engelstad_nitro.csv');d=d[d.nitro>=201]
print(int(d.groupby('year')['yield'].mean().idxmax()))