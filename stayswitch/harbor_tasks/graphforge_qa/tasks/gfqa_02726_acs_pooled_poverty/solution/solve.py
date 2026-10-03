import os,json
import pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
j=json.load(open(D+'/census_acs_oilgas_states.json'))
E=[v['B17001']['estimate'] for v in j['data'].values()]
p=100*sum(e['B17001002'] for e in E)/sum(e['B17001001'] for e in E)
m=max(100*e['B17001002']/e['B17001001'] for e in E)
print(round(m-p,2))
