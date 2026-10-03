import os, pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
a=pd.read_csv(D+'/2025-Table_3A_Supervisory_Severely_Adverse_Domestic.csv');a=a[a.Date.str.startswith('2026')];print(round(a['10-year Treasury yield'].mean()-a['3-month Treasury rate'].mean(),2))
