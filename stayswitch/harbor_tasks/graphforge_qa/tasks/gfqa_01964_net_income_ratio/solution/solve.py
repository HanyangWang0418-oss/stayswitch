import os,pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
f=pd.read_csv(D+'/fdic_fitb_financials.csv')
q=pd.read_csv(D+'/peer_bank_credit_metrics.csv')
x=f[f.REPDTE==20231231].NETINC.iloc[0]
print(round(float(x/q.NETINC.dropna().median()),2))