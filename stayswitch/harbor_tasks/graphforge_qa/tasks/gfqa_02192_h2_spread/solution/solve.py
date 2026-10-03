import os,pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
d=pd.read_csv(D+'/treasury_yield_curve_2023.csv')
d['Date']=pd.to_datetime(d['Date'])
h=d[d.Date.dt.month>=7]
print(round(float((h['10 Yr']-h['3 Mo']).mean()),2))