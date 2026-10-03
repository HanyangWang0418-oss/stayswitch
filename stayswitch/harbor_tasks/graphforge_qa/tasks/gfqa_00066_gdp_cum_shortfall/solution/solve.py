import os,json
import pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
a=pd.read_csv(D+'/2024_severely_adverse_domestic.csv');b=pd.read_csv(D+'/2024_supervisory_baseline_domestic.csv')
f=lambda d:d[d.Date.str.startswith('2025')]['Real GDP growth'].sum()
print(round(f(b)-f(a),1))
