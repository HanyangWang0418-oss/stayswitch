import os, pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
a=pd.read_csv(D+'/treasury_yields_2026.csv');a['s']=a['30 Yr']-a['2 Yr'];a['d']=pd.to_datetime(a['Date'],format='%m/%d/%Y')
m=a['s'].max();print(a[a.s>m-1e-9].sort_values('d').iloc[0]['Date'])
