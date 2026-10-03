import os,json
import pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
s=pd.read_csv(D+'/sequencing_costs_datahub.csv')
s['ch']=s['Cost per Genome'].pct_change()
print(s.loc[s.ch.idxmin(),'Date'])
