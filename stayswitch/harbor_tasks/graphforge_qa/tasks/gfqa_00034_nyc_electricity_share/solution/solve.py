import os,csv,json
D=os.environ.get('DATA_DIR','/app/data')
r=list(csv.DictReader(open(D+'/nyc_municipal_ghg_inventory.csv')))
t=[float(x['FY 2024 tCO2e']) for x in r if x['Main Sector']=='Total'][0]
e=sum(float(x['FY 2024 tCO2e']) for x in r if x['Source']=='Electricity' and x['Main Sector']!='Total')
print(round(e/t*100,2))
