import os,json,pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
t=json.loads(json.load(open(D+'/fdic_jpmc_financials.json'))['text'])
d=pd.DataFrame([x['data'] for x in t['data']]).sort_values('REPDTE')
print(round((d.ASSET.iloc[-1]/d.ASSET.iloc[0]-1)*100,2))