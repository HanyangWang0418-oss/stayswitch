import os,pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
d=pd.read_csv(D+'/nyc_population_projections_2010_2040.csv')
d['Borough']=d.Borough.str.strip();d['Age Group']=d['Age Group'].str.strip()
d=d[d.Borough!='New York City']
s=d[d['Age Group'].str.startswith('School')]['2040'].sum();t=d[d['Age Group']=='Total']['2040'].sum()
print(round(float(100*s/t),2))