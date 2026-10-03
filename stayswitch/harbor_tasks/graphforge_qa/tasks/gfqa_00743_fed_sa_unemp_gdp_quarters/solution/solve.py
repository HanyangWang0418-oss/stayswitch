import os, pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
a=pd.read_csv(D+'/2025-Table_3A_Supervisory_Severely_Adverse_Domestic.csv').reset_index(drop=True);i=a['Unemployment rate'].idxmax();print(round(a.loc[:i,'Real GDP growth'].sum(),1))
