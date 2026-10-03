import os, pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
a=pd.read_csv(D+'/uc_davis_complications_deaths.csv');b=pd.read_csv(D+'/national_complications_deaths_benchmarks.csv')
m=a.merge(b[['measure_id','national_rate']],on='measure_id');m=m[m.measure_id.str.startswith('PSI_')&~m.measure_id.isin(['PSI_90','PSI_04'])]
print(int((m.score>m.national_rate).sum()))
