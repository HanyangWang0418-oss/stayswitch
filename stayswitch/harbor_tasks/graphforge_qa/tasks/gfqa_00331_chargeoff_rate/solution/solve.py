import os,json
import pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
l=pd.read_csv(D+'/rbtc_12q_loan_portfolio.csv',dtype={'REPDTE':str})
c=pd.read_csv(D+'/rbtc_12q_capital_ratios.csv',dtype={'REPDTE':str})
m=l.merge(c[['ID']],on='ID')
m=m[m.REPDTE.str.startswith('2025')]
print(round(100*m.NTLNLS.sum()/m.LNLSNET.mean(),3))
