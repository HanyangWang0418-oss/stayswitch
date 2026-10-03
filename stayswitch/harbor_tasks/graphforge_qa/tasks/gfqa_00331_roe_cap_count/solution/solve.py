import os,json
import pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
l=pd.read_csv(D+'/rbtc_12q_loan_portfolio.csv')
c=pd.read_csv(D+'/rbtc_12q_capital_ratios.csv')
m=l.merge(c[['ID','ROE']],on='ID')
print(int(((m.ROE>15)&(m.NTLNLSR<0.9)).sum()))
