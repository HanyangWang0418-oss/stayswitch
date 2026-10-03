import os,json,pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
d=pd.read_csv(D+'/city_owned_vacant_lots.csv')
d=d[d['Platted (Y/N)'].astype(str)=='True']
s=d.groupby(d.Zoning.str.split('/').str[0])['Lot Size (Sqft)'].sum()
print(round(s.max()/43560,2))