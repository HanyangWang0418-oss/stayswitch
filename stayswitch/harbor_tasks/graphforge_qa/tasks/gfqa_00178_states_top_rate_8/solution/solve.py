import os,csv,json
D=os.environ.get('DATA_DIR','/app/data')
rows=[x for x in csv.reader(open(D+'/state_corporate_tax_rates_2024.csv'))][1:]
import re
m={}
for x in rows:
    if x[1].strip():
        n=re.sub(r'\s*\(.*\)','',x[0]).strip();m[n]=max(m.get(n,0),float(x[1].rstrip('%')))
print(sum(1 for v in m.values() if v>8.0))
