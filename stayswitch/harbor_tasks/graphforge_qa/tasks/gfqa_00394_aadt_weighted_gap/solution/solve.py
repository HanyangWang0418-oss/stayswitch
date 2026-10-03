import os,pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
d=pd.read_csv(D+'/fdot-aadt-us19-sr55-pasco-hernando.csv')
d['L']=d.END_POST-d.BEGIN_POST
w=lambda c:(d[d.COUNTY==c].AADT*d[d.COUNTY==c].L).sum()/d[d.COUNTY==c].L.sum()
print(round(float(w('Pasco')-w('Hernando'))))