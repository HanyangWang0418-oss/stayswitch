import os,csv,json
D=os.environ.get('DATA_DIR','/app/data')
r=list(csv.DictReader(open(D+'/fdic_financials_enb.csv')))
v=[float(x['EQ'])/float(x['ASSET'])*100 for x in r if x['REPDTE'][4:]=='1231' and '2019'<=x['REPDTE'][:4]<='2024']
assert len(v)==6
print(round(sum(v)/6,2))
