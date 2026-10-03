import os,json
import pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
r=sorted([x['data'] for x in json.load(open(D+'/bofa_fdic_sdi_financials.json'))['data']],key=lambda d:d['REPDTE'])
a,b=r[0],r[-1]
print(round(100*(b['LIAB']/a['LIAB']-1)-100*(b['EQTOT']/a['EQTOT']-1),2))
