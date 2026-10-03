import os,csv,json
D=os.environ.get('DATA_DIR','/app/data')
p=list(csv.DictReader(open(D+'/sf_parcels_block3701.csv')))
l={x['mapblklot']:x for x in csv.DictReader(open(D+'/sf_landuse_block3701.csv'))}
c=[x['mapblklot'] for x in p if x['street_name'].strip() and x['mapblklot'] in l]
print(max(c,key=lambda k:float(l[k]['total_comm'])))
