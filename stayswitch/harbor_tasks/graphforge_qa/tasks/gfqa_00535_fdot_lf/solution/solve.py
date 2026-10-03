import os,json
import pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
d=pd.read_csv(D+'/fdot_biditems_e8v40.csv',dtype=str)
d['Quantity']=d.Quantity.astype(float)
l=d[d.Unit.str.strip()=='LF']
s=l[l.ShortDesc.str.contains('SIGN|BARRIER')].Quantity.sum()
print(round(s+(l.Quantity>1000).sum(),1))
