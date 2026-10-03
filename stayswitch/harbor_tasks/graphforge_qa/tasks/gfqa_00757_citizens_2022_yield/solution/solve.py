import os, pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
f=pd.read_csv(D+'/citizens_bank_financials.csv');g=pd.read_csv(D+'/citizens_bank_full_financials.csv')
i=g[g.REPDTE==20221231].INTINC.iloc[0];a=f[f.REPDTE.isin([20220331,20220630,20220930,20221231])].ASSET.mean();print(round(i/a*100,2))
