import os,pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
d=pd.read_csv(D+'/treasury_yield_curve_2023.csv')
print(int(((d['2 Yr']-d['10 Yr'])>=0.5-1e-9).sum()))