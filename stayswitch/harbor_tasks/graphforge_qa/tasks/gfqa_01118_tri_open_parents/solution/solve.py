import os,json,pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
t=pd.concat([pd.read_csv(D+'/tri_facilities_st_john_baptist.csv'),pd.read_csv(D+'/tri_facilities_st_james_parish.csv')])
t=t[(t.fac_closed_ind==0)&t.standardized_parent_company.notna()]
print(t.standardized_parent_company.nunique())