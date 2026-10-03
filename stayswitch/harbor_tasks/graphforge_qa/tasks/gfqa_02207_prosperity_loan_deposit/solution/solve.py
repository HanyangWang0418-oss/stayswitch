import os,csv,json
D=os.environ.get('DATA_DIR','/app/data')
d=[x['data'] for x in json.load(open(D+'/prosperity_bank_recent_quarters.json'))['data']]
print(min(d,key=lambda x:x['LNLSNET']/x['DEP'])['REPDTE'])
