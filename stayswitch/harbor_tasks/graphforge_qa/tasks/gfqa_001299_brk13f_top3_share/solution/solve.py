import os, pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
a=pd.read_csv(D+'/berkshire_13f_q4_2024_holdings.csv');g=a.groupby('nameOfIssuer').valueUSD.sum().sort_values(ascending=False)
print(round(g.head(3).sum()/a.valueUSD.sum()*100,2))
