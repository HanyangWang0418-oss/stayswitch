import os, pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
a=pd.read_csv(D+'/uc_davis_complications_deaths.csv');b=pd.read_csv(D+'/national_complications_deaths_benchmarks.csv')
m=a.merge(b[['measure_id','national_rate']],on='measure_id');m=m[m.measure_id.str.startswith('MORT_30')]
print(round((m.national_rate-m.score).mean(),2))
