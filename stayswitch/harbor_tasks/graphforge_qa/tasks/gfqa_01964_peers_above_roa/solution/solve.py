import os,pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
f=pd.read_csv(D+'/fdic_fitb_financials.csv')
q=pd.read_csv(D+'/peer_bank_credit_metrics.csv')
r=f[f.REPDTE==20231231].ROA.iloc[0]
print(int((q.ROA>r).sum()))