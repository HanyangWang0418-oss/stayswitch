import os, pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
a=pd.read_csv(D+'/locations.csv',keep_default_na=False);u=a[a.abbreviation=='US'].count_rate3.iloc[0];print(int(a[a.abbreviation!='US'].count_rate3.sum()-u))
