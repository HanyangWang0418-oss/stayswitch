import os,json
import pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
f=lambda y:json.load(open(D+'/stanford_cost_report_%d.json'%y))
g=lambda d:float(d['Total Costs'])/float(d['Total Bed Days Available'])
print(round(100*(g(f(2023))/g(f(2022))-1),2))
