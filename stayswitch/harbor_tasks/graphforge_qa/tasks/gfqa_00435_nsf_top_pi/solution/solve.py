import os,json,pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
d=pd.read_csv(D+'/nsf_developmental_sciences_awards.csv')
d['y']=pd.to_datetime(d.startDate,format='%m/%d/%Y').dt.year
print(d[d.y>=2010].groupby('piLastName').fundsObligatedAmt.sum().idxmax())