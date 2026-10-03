import os,json
import pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
f=lambda y:json.load(open(D+'/stanford_cost_report_%d.json'%y))
g=lambda d:100*float(d['Net Income'])/(float(d['Net Patient Revenue'])+float(d['Total Other Income']))
print(round(g(f(2023))-g(f(2022)),2))
