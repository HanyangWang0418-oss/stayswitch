import os,json
import pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
j=json.load(open(D+'/bls_employment_api_jul2026.json'))
s=[x for x in j['Results']['series'] if x['seriesID']=='CES0000000001'][0]['data']
v=sorted([(x['year']+'-'+x['period'][1:],float(x['value'])) for x in s])
i=min(range(1,len(v)),key=lambda i:v[i][1]-v[i-1][1])
print(v[i][0])
