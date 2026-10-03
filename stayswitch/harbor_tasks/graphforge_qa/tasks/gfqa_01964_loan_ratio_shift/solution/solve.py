import os,pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
f=pd.read_csv(D+'/fdic_fitb_financials.csv').set_index('REPDTE')
r=lambda k:100*f.loc[k,'LNLSNET']/f.loc[k,'ASSET']
print(round(float(r(20251231)-r(20211231)),2))