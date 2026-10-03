import os, pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
a=pd.read_csv(D+'/cooke_county_grade_crossing_aadt.csv').dropna(subset=['aadt']);g=a[a.cityname=='GAINESVILLE'].aadt.mean();o=a[a.cityname!='GAINESVILLE'].aadt.mean();print(round(g-o,2))
