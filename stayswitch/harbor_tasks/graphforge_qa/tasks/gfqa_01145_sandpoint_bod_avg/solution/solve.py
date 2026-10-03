import os,csv,json
D=os.environ.get('DATA_DIR','/app/data')
d=json.load(open(D+'/sandpoint_echo_dmr_summary.json'))['monitoring_summary']
def m(k):
    v=[float(x['value']) for x in d[k]['sample_values'] if x['statistic']=='MO AVG'];return sum(v)/len(v)
e=m('BOD, 5-day, 20 deg. C [Effluent Gross]');i=m('BOD, 5-day, 20 deg. C [Raw Sewage Influent]')
print(round((1-e/i)*100,1))
