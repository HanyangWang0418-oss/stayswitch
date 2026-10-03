import os,json
import pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
a=pd.read_csv(D+'/2024_severely_adverse_domestic.csv');b=pd.read_csv(D+'/2024_supervisory_baseline_domestic.csv')
m=a.merge(b,on='Date',suffixes=('_a','_b'))
r=m.loc[m['Unemployment rate_a'].idxmax()]
print(round(r['Unemployment rate_a']-r['Unemployment rate_b'],1))
