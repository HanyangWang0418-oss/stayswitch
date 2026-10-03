import os,json,pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
a=pd.read_csv(D+'/tri_facilities_st_john_baptist.csv');b=pd.read_csv(D+'/tri_facilities_st_james_parish.csv')
f=lambda d:(d.fac_closed_ind==1).mean()*100
print(round(f(a)-f(b),2))