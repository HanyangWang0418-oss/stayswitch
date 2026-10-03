import os,json
import pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
g=pd.read_csv(D+'/gsa_pov_mileage_rates.csv')
r=g.loc[g.airplane.idxmax()]
print(round(r.automobile/r.moving,3))
