import os,pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
d=pd.read_csv(D+'/dfast_2024_severely_adverse_scenario.csv')
print(round(float(d[d['10-year Treasury yield']<=1.0]['Unemployment rate'].mean()),2))