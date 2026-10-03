import os,csv,json
D=os.environ.get('DATA_DIR','/app/data')
s={}
for f in ['toyota_camry_2018_recalls.json','honda_accord_2018_recalls.json','ford_f150_2018_recalls.json']:
    for r in json.load(open(D+'/'+f))['results']:
        y=int(r['ReportReceivedDate'][-4:])
        if 2018<=y<=2020 and r['Component'].startswith(('FUEL SYSTEM','ENGINE','POWER TRAIN')): s[r['NHTSACampaignNumber']]=1
print(len(s))
