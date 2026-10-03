import os,csv,json
D=os.environ.get('DATA_DIR','/app/data')
p=list(csv.DictReader(open(D+'/sf_parcels_block3701.csv')))
l={x['mapblklot']:x for x in csv.DictReader(open(D+'/sf_landuse_block3701.csv'))}
a=[l[x['mapblklot']] for x in p if x['active']=='true' and x['mapblklot'] in l]
print(round(sum(float(x['total_comm']) for x in a)/sum(float(x['resunits']) for x in a),2))
