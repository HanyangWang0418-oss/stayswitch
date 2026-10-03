import os,csv,json
D=os.environ.get('DATA_DIR','/app/data')
n=0;k=0
for f in ['toyota_camry_2018_recalls.json','honda_accord_2018_recalls.json','ford_f150_2018_recalls.json']:
    for r in json.load(open(D+'/'+f))['results']:
        n+=1;k+=int(r['ReportReceivedDate'][-4:])>=2021
print(round(k/n*100,1))
