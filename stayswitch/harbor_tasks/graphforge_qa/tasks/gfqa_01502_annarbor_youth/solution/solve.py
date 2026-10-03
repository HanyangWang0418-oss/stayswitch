import os,json
import pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
j=json.load(open(D+'/census_demographics.json'))
e=list(j['data'].values())[0]['B01001']['estimate']
s=sum(e['B01001%03d'%i] for i in list(range(7,11))+list(range(31,35)))
print(round(100*s/e['B01001001'],2))
