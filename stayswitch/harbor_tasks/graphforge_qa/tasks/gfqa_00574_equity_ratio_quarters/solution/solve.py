import os,json
import pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
def L(f):
    return {r['data']['REPDTE']:r['data'] for r in json.load(open(D+'/'+f))['data']}
b=L('ffiec_bofa_financials.json');j=L('ffiec_jpmorgan_capital_data.json')
n=0
for k in set(b)&set(j):
    if j[k]['EQ']/j[k]['ASSET']>b[k]['EQ']/b[k]['ASSET']: n+=1
print(n)
