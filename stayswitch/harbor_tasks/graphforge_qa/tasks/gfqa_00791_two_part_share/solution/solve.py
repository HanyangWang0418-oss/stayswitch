import os,json
import pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
c=pd.read_csv(D+'/caltrans_2020_adjusted_urban_area.csv')
s=c[(c.UrbanAreas==2)&(c.Area_sqm>50)].Population.sum()
print(round(100*s/c.Population.sum(),2))
