import os, pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
a=pd.read_csv(D+'/uc_davis_hcahps_scores.csv');b=pd.read_csv(D+'/state_hcahps_averages_ca.csv')
a=a[a.hcahps_measure_id.str.endswith('_A_P')][['hcahps_measure_id','hcahps_answer_percent']];b=b[b.hcahps_measure_id.str.endswith('_A_P')][['hcahps_measure_id','hcahps_answer_percent']]
m=a.merge(b,on='hcahps_measure_id',suffixes=('_u','_s'))
for c in ['hcahps_answer_percent_u','hcahps_answer_percent_s']: m[c]=pd.to_numeric(m[c],errors='coerce')
m=m.dropna();print(round((m.hcahps_answer_percent_u-m.hcahps_answer_percent_s).mean(),2))
