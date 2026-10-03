import os,json
import pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
s=pd.read_csv(D+'/sequencing_costs_datahub.csv')
a=s[s.Date.str.startswith('2008')]['Cost per Genome'].mean();b=s[s.Date.str.startswith('2012')]['Cost per Genome'].mean()
print(round(a/b,2))
