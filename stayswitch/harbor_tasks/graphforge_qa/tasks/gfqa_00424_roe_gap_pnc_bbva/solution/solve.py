import os,json,pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
def L(f):
    return pd.DataFrame([x['data'] for x in json.load(open(D+'/'+f))['data']])
p=L('pnc_bank_financials_fdic.json');b=L('bbva_usa_financials_fdic.json')
a=p[p.REPDTE.str[:4]=='2025'].ROE.mean();c=b[b.REPDTE.str[:4]=='2019'].ROE.mean()
print(round(a-c,2))