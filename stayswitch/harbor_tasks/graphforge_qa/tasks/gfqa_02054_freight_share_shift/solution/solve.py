import os,json,pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
d=pd.read_csv(D+'/bts_freight_ton_miles.csv')
def s(y):
    x=d[d.Year==y].set_index('Mode').Value
    return (x['Rail']+x['Pipeline']+x['Water'])/x['Total']*100
print(round(s(2022)-s(2017),2))