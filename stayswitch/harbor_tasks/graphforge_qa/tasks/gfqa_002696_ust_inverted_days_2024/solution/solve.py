import os, pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
a=pd.read_csv(D+'/treasury_yields_2024.csv')
print(int(((a['3 Mo']>a['10 Yr'])&(a['2 Yr']>a['10 Yr'])).sum()))
