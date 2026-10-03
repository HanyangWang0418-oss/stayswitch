import os,json,pandas as pd
D=os.environ.get('DATA_DIR','/app/data')
d=json.load(open(D+'/web_server_benchmark_2023.json'))
b=lambda s:float(s.replace(',','').replace('mb','').strip())
print(max(d,key=lambda x:b(x['bytes'])/x['requests'])['server'])