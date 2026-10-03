import os,json,pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
d=json.load(open(D+'/tsic_propublica_990_financials.json'))['filings_with_data']
r=lambda a,b:sum(f['totliabend']/f['totassetsend']*100 for f in d if a<=f['tax_prd_yr']<=b)/sum(1 for f in d if a<=f['tax_prd_yr']<=b)
print(round(r(2018,2023)-r(2010,2015),2))