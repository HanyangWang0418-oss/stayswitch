import os,csv,json
D=os.environ.get('DATA_DIR','/app/data')
r=list(csv.DictReader(open(D+'/marketing_campaign_metrics.csv')))
r=[x for x in r if x['campaign_name'].lower().startswith('facebook')]
print(round(sum(float(x['mark_spent']) for x in r)/sum(float(x['orders']) for x in r),2))
