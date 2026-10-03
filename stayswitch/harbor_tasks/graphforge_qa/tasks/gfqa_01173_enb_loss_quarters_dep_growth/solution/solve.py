import os,csv,json
D=os.environ.get('DATA_DIR','/app/data')
r=list(csv.DictReader(open(D+'/fdic_financials_enb.csv')))
q=[x for x in r if '20230331'<=x['REPDTE']<='20260331']
n=sum(1 for x in q if float(x['NETINC'])<0)
d={x['REPDTE']:float(x['DEP']) for x in r}
g=(d['20260331']/d['20230331']-1)*100
print(round(n*g,2))
