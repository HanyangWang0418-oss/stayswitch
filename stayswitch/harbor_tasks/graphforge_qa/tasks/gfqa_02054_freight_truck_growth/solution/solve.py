import os,json,pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
d=pd.read_csv(D+'/bts_freight_ton_miles.csv')
d['Mode']=d.Mode.replace({'Truck':'Highway'})
g=lambda y,m:float(d[(d.Year==y)&(d.Mode==m)].Value.iloc[0])
print(round((g(2050,'Highway')/g(2019,'Highway')-1)*100-g(2050,'Highway')/g(2050,'Total')*100,2))