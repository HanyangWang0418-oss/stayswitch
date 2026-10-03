import os,pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
d=pd.read_csv(D+'/treasury_yield_curve_2023.csv')
d['g']=d['3 Mo']-d['10 Yr']
d['dt']=pd.to_datetime(d['Date'])
m=d[d.g.round(6)==d.g.round(6).max()].sort_values('dt').iloc[-1]
print(m['Date'])