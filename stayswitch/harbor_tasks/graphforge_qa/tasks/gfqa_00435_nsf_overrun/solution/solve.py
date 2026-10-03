import os,json,pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
d=pd.read_csv(D+'/nsf_developmental_sciences_awards.csv')
e=(d.fundsObligatedAmt-d.estimatedTotalAmt).clip(lower=0).sum()
print(round(e/d.estimatedTotalAmt.sum()*100,2))