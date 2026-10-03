import os,json,pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
d=pd.read_csv(D+'/landsat_l1_scenes_export.csv').dropna(subset=['eo_cloud_cover'])
m=d.groupby('platform').eo_cloud_cover.mean()
print(round(m['LANDSAT_9']-m['LANDSAT_8'],2))