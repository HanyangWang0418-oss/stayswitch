import os, pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
a=pd.read_csv(D+'/epa_aqi_by_county_2021_in_ky.csv');a=a[a['Days with AQI']>=360].copy();a['r']=a['Days PM2.5']/a['Days with AQI'];r=a.sort_values('r',ascending=False).iloc[0];print(r.County+', '+r.State)
