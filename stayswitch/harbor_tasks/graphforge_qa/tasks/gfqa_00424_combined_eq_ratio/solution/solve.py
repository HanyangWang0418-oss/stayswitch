import os,json,pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
def L(f):
    return pd.DataFrame([x['data'] for x in json.load(open(D+'/'+f))['data']])
p=L('pnc_bank_financials_fdic.json');b=L('bbva_usa_financials_fdic.json')
d=max(set(p.REPDTE)&set(b.REPDTE))
P=p[p.REPDTE==d].iloc[0];B=b[b.REPDTE==d].iloc[0]
print(round((P.EQ+B.EQ)/(P.ASSET+B.ASSET)*100,2))