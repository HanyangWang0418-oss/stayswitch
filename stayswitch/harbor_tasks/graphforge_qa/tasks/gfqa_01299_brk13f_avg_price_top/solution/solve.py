import os, pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
a=pd.read_csv(D+'/berkshire_13f_q4_2024_holdings.csv');g=a.groupby('nameOfIssuer')[['shares','valueUSD']].sum();g=g[g.shares>50e6];print((g.valueUSD/g.shares).idxmax())
