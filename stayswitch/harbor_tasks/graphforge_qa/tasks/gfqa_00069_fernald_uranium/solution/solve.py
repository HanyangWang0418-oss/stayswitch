import os,csv,json
D=os.environ.get('DATA_DIR','/app/data')
r=list(csv.DictReader(open(D+'/wqp_fernald_uranium_results.csv')))
s={x['MonitoringLocationIdentifier'] for x in r if x['ResultMeasureValue'].strip() and x['ResultMeasure/MeasureUnitCode']=='ug/l' and float(x['ResultMeasureValue'])>2.0}
print(len(s))
