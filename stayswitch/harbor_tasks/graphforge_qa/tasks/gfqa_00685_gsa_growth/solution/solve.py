import os,json
import pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
g=pd.read_csv(D+'/gsa_pov_mileage_rates.csv').set_index('effective_date')
print(round(100*(g.loc['2026-07-01','automobile']/g.loc['2022-01-01','automobile']-1),2))
