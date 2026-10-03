import os,pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
d=pd.read_csv(D+'/williams_ghgrp_emissions_2023.csv')
d=d[d['Industry Type (subparts)'].str.contains('W-PROC',na=False)]
print(round(float(100*d['CH4 emissions'].sum()/d['Total reported direct emissions (mtCO2e)'].sum()),2))