import os, pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
a=pd.read_csv(D+'/locations.csv',keep_default_na=False);u=a[a.abbreviation=='US'].population.iloc[0];s=a[a.abbreviation!='US'].population.nlargest(5).sum();print(round(s/u*100,2))
