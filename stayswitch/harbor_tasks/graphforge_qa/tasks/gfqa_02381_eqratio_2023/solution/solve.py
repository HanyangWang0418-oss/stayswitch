import os,json,pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
r=pd.read_csv(D+'/republic_bank_full_financials.csv',dtype={'REPDTE':str});m=pd.read_csv(D+'/peer_mid_penn_bank_financials.csv',dtype={'REPDTE':str})
f=lambda d:(d[d.REPDTE.str[:4]=='2023'].eval('EQ/ASSET*100')).mean()
print(round(f(m)-f(r),2))