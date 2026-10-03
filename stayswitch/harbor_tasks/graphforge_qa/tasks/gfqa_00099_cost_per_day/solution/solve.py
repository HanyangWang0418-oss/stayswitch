import os,pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
d=pd.read_csv(D+'/sparcs_sample_records.csv')
d=d[(d.health_service_area=='New York City')&(d.age_group=='0-17')]
print(round(float(d.total_costs.sum()/d.length_of_stay.sum()),2))