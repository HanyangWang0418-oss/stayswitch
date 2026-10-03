import os,csv,json
D=os.environ.get('DATA_DIR','/app/data')
rows=[x for x in csv.reader(open(D+'/state_corporate_tax_rates_2024.csv'))][1:]
b=[(float(x[3].replace('$','').replace(',','')),float(x[1].rstrip('%'))/100) for x in rows if x[0].startswith('Louisiana')]
b.sort();inc=400000;t=0
for i,(lo,r) in enumerate(b):
    hi=b[i+1][0] if i+1<len(b) else float('inf')
    t+=max(0,min(inc,hi)-lo)*r
print(round(t,2))
