import os,pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
d=pd.read_csv(D+'/fia_state_aboveground_carbon_estimates.csv')
d=d[d.SE_PERCENT<1.5]
print(d.loc[(d.ESTIMATE/d.PLOT_COUNT).idxmax(),'STATE_NAME'])