import os,csv,json
D=os.environ.get('DATA_DIR','/app/data')
r=list(csv.DictReader(open(D+'/texas_facility_136975_inspections.csv')))
r=[x for x in r if x['activity_type']=='INSPECTION' and '2022'<=x['activity_date'][:4]<='2024']
print(round(sum(x['violation_found']=='Yes' for x in r)/len(r)*100,1))
