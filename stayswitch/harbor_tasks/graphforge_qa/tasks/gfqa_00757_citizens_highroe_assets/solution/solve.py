import os, pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
f=pd.read_csv(D+'/citizens_bank_financials.csv');g=pd.read_csv(D+'/citizens_bank_full_financials.csv')
d=f[f.ROE>15].REPDTE
print(round(g[g.REPDTE.isin(d)].LIAB.mean()-f[f.REPDTE.isin(d)].EQ.mean(),2))
