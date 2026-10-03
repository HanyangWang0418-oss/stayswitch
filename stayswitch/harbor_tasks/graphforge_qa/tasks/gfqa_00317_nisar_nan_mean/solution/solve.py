import os,csv,json
D=os.environ.get('DATA_DIR','/app/data')
r=list(csv.DictReader(open(D+'/NISAR_GUNW_QA_SUMMARY.csv')))
v=[float(x['Actual']) for x in r if x['Check']=='% NaN pixels under threshold?' and 'unwrappedInterferogram' in x['Notes']]
print(round(sum(v)/len(v),2))
