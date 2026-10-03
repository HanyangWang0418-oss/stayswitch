import os,pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
d=pd.read_csv(D+'/watershed_reach_integrity_scores.csv')
d=d[(d['Year of Observation']==2021)&d.Overall.notna()]
g=d.groupby('Watershed Name').Overall.agg(['mean','count'])
g=g[g['count']>=3]
print(g['mean'].idxmax())