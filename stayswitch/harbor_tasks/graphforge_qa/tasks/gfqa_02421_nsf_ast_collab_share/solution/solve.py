import os, pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
a=pd.read_csv(D+'/nsf_ast_awards_summary.csv');print(round(a[a.Title.str.startswith('Collaborative Research')].Amount.sum()/a.Amount.sum()*100,2))
