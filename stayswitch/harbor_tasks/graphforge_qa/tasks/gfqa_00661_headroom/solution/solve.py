import os,pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
import json
d=pd.read_csv(D+'/mdlg_violation_items.csv')
s=json.load(open(D+'/osha_penalty_schedule_2024.json'))['violation_types']
d=d[d.alj_assessed_penalty>0]
m=d.citation_type.str.lower().map(lambda k:s[k]['statutory_max_per_violation'])
print(int((m-d.alj_assessed_penalty).sum()))