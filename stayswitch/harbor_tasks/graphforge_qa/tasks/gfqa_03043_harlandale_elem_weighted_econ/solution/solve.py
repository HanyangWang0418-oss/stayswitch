import os, pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
a=pd.read_csv(D+'/harlandale_enrollment_2025.csv');a=a[a['Campus Number'].notna()&(a['School Type']=='Elementary')];print(round((a['% Economically Disadvantaged']*a['Number of Students']).sum()/a['Number of Students'].sum()*100,2))
