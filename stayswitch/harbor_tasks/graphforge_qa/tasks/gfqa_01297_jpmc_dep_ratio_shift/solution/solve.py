import os,json,pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
t=json.loads(json.load(open(D+'/fdic_jpmc_financials.json'))['text'])
d=pd.DataFrame([x['data'] for x in t['data']]).sort_values('REPDTE')
r=(d.DEP/d.ASSET*100)
print(round(r.iloc[-1]-r.iloc[0],2))