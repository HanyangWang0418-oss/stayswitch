import os,csv,json
D=os.environ.get('DATA_DIR','/app/data')
d={x['data']['REPDTE']:x['data'] for x in json.load(open(D+'/prosperity_bank_recent_quarters.json'))['data']}
print(round(d['20260331']['NETINC']*4/((d['20260331']['ASSET']+d['20250331']['ASSET'])/2)*100,2))
