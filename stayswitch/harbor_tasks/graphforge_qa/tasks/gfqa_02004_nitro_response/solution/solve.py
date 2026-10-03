import os,json,pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
d=pd.read_csv(D+'/engelstad_nitro.csv');k=d[d['loc']=='Knoxville']
p=k.pivot_table(index='year',columns='nitro',values='yield')
print(round((p[335]-p[0]).mean(),2))