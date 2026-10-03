import os,json
import pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
c=pd.read_csv(D+'/caltrans_2020_adjusted_urban_area.csv')
c=c[c.Population>=100000].copy()
c['d']=c.Population/c.Area_sqm
print(c.sort_values('d').iloc[-1].NAME)
