import os,json
import pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
j=json.load(open(D+'/bls_employment_api_jul2026.json'))
S={x['seriesID']:{(d['year'],d['period']):float(d['value']) for d in x['data']} for x in j['Results']['series'] if x['seriesID'].startswith('CES')}
k=[('2026','M0%d'%i) for i in range(1,8)]
print(round(sum(100*S['CES0500000001'][m]/S['CES0000000001'][m] for m in k)/7,3))
