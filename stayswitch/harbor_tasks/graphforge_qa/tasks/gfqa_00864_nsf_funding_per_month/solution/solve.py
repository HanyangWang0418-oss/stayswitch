import os,csv,json
D=os.environ.get('DATA_DIR','/app/data')
def f(n):
    a=json.load(open(D+'/'+n))['response']['award'][0]
    s=a['startDate'].split('/');e=a['expDate'].split('/')
    m=12*(int(e[2])-int(s[2]))+int(e[0])-int(s[0])
    return float(a['estimatedTotalAmt'])/m
print(round(f('nsf_award_1901630.json')/f('nsf_award_1103235_confuser.json'),2))
