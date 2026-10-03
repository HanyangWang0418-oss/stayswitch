import os,pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
d=pd.read_csv(D+'/fdot-aadt-us19-sr55-pasco-hernando.csv')
d=d[(d.COUNTY=='Pasco')&(d.AADT>=5000)]
print(round(float((d.AADT*d.TFCTR/100).sum()),1))