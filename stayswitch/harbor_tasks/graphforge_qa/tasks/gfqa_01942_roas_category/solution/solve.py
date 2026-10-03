import os,csv,json
D=os.environ.get('DATA_DIR','/app/data')
import collections
r=list(csv.DictReader(open(D+'/marketing_campaign_metrics.csv')))
s=collections.defaultdict(float);v=collections.defaultdict(float)
for x in r:
    s[x['category']]+=float(x['mark_spent']);v[x['category']]+=float(x['revenue'])
print(max(s,key=lambda k:v[k]/s[k]))
