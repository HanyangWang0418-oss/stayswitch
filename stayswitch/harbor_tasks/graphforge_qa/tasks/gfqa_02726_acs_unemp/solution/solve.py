import os,json
import pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
j=json.load(open(D+'/census_acs_oilgas_states.json'))
r={g:v['B23025']['estimate']['B23025005']/v['B23025']['estimate']['B23025003'] for g,v in j['data'].items()}
print(j['geography'][max(r,key=r.get)]['name'])
