import os,json,pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
d=json.load(open(D+'/tsic_propublica_990_financials.json'))['filings_with_data']
tot=0
for f in d:
    if 2014<=f['tax_prd_yr']<=2023:
        s=f['totrevenue']-f['totfuncexpns']
        if s<0: tot+=s
print(tot)