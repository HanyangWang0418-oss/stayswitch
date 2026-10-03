import os,pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
d=pd.read_csv(D+'/fia_state_aboveground_carbon_estimates.csv')
print(round(float(100*d[d.PLOT_COUNT>=3000].ESTIMATE.sum()/d.ESTIMATE.sum()),2))