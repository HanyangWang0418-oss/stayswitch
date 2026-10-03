import os,json,pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
r=pd.read_csv(D+'/republic_bank_full_financials.csv',dtype={'REPDTE':str});m=pd.read_csv(D+'/peer_mid_penn_bank_financials.csv',dtype={'REPDTE':str})
j=r.merge(m,on='REPDTE',suffixes=('_r','_m'))
print(int((j.ROE_m>j.ROE_r).sum()))