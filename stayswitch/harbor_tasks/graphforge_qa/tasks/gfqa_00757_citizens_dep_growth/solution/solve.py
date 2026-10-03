import os, pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
f=pd.read_csv(D+'/citizens_bank_financials.csv').set_index('REPDTE');print(round((f.loc[20230930,'DEP']/f.loc[20220930,'DEP']-1)*100,2))
