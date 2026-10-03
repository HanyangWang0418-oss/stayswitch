import os,json
import pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
r=sorted([x['data'] for x in json.load(open(D+'/bofa_fdic_sdi_financials.json'))['data']],key=lambda d:d['REPDTE'])
best=min(range(1,len(r)),key=lambda i:r[i]['ASSET']-r[i-1]['ASSET'])
print(r[best]['REPDTE'])
