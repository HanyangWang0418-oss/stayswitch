import os,pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
d=pd.read_csv(D+'/watershed_reach_integrity_scores.csv')
d=d.dropna(subset=['Overall','Habitat','Contact Recreation'])
print(int(((d.Habitat-d['Contact Recreation'])>=20).sum()))