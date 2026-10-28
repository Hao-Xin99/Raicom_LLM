

#用python文件充当客户端
import requests

Base='http://127.0.0.1:8700'

#客户端,GET的参数用的是params
r=requests.get(url=f'{Base}/name',params={'name':'小明'},timeout=5)
print ('GET:',r.text,r.status_code,'\n')


#对于post用的是json
r_2=requests.post(url=f'{Base}/information',json={'name':'小红','ID':'2025','class':'4'},timeout=5)
print('POST:',r_2.json(),r_2.status_code)






