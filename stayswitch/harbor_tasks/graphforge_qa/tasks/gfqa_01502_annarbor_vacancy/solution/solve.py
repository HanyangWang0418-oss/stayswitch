import os,json
import pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
j=json.load(open(D+'/census_demographics.json'))
d=list(j['data'].values())[0]
h=d['B25001']['estimate']['B25001001'];o=d['B25003']['estimate']['B25003002'];r=d['B25003']['estimate']['B25003003']
print(round(((h-o-r)/h)/(r/(o+r)),4))
