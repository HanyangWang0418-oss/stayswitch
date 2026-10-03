import os, pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
a=pd.read_csv(D+'/epa_aqi_by_county_2021_in_ky.csv');p=a.groupby('State')['Max AQI'].apply(lambda s:(s>=101).mean()*100);print(round(abs(p.max()-p.min()),2))
