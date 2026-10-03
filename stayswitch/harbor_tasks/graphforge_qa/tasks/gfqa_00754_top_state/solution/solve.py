import os,pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
d=pd.read_csv(D+'/williams_ghgrp_emissions_2023.csv')
d=d[d['Industry Type (sectors)']!='Power Plants']
print(d.groupby('State')['Total reported direct emissions (mtCO2e)'].sum().idxmax())