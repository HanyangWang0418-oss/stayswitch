import os,csv,json
D=os.environ.get('DATA_DIR','/app/data')
r=list(csv.DictReader(open(D+'/nyc_municipal_ghg_inventory.csv')))
b=[x for x in r if x['Main Sector']=='Buildings']
a=sum(float(x['FY 2006 tCO2e']) for x in b);c=sum(float(x['FY 2024 tCO2e']) for x in b)
print(round((c/a-1)*100,2))
