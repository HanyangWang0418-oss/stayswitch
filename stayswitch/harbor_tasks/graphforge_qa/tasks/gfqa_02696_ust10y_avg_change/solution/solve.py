import os, pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
a=pd.read_csv(D+'/treasury_yields_2024.csv');b=pd.read_csv(D+'/treasury_yields_2026.csv')
print(round(b['10 Yr'].mean()-a['10 Yr'].mean(),2))
