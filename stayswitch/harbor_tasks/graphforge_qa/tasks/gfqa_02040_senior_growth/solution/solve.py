import os,pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
d=pd.read_csv(D+'/nyc_population_projections_2010_2040.csv')
d['Borough']=d.Borough.str.strip();d['Age Group']=d['Age Group'].str.strip()
x=d[(d['Age Group']=='65 and over')&(d.Borough!='New York City')].copy()
x['g']=x['2040']/x['2010']
print(x.loc[x.g.idxmax(),'Borough'])