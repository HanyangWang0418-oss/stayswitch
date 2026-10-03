import os,pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
d=pd.read_csv(D+'/sparcs_sample_records.csv')
d=d[d.length_of_stay<=3]
print(d.groupby('facility_name').total_costs.sum().idxmax())