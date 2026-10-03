import os, pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
a=pd.read_csv(D+'/epa_aqi_by_county_2021_in_ky.csv');a=a[a.State=='Kentucky'];print(round(a['Good Days'].sum()/a['Days with AQI'].sum()*100,2))
