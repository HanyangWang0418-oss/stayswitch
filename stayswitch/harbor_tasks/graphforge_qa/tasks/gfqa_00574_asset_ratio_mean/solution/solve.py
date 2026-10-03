import os,json
import pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
def L(f):
    return {r['data']['REPDTE']:r['data'] for r in json.load(open(D+'/'+f))['data']}
b=L('ffiec_bofa_financials.json');j=L('ffiec_jpmorgan_capital_data.json')
k=sorted(set(b)&set(j))
print(round(sum(j[x]['ASSET']/b[x]['ASSET'] for x in k)/len(k),3))
