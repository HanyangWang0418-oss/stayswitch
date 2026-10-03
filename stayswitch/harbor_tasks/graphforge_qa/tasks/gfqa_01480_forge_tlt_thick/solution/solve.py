import os,json,pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
d=pd.read_csv(D+'/FORGE_Well_Lith_Logs_2018.csv').dropna(subset=['Well'])
s=d[d.Unit=='Tlt'].groupby('Well').thick_ft.sum().sort_values(ascending=False)
print(round(float(s.iloc[0]-s.iloc[1]),2))