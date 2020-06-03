#-*- codong:utf-8 -*-
import numpy as np
import pandas as pd
import math
import os
import datetime
import time
import subprocess
#import DLTYC.py
print('预测程序挂载成功')
def ONE():
    print('进程开始：'+datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('数据更新中……')
    f=open(r'C:/Users/Administrator/Desktop/DLT/DLT2.txt','r',encoding='utf-8')
    DLT1=f.read()
    f.close
    import re
    A = re.split(r'\n',DLT1)                             #num
    DLT2=[]
    for a in A:
        B=re.split(r' ',a)
        DLT2.append(B[0])
    #print(DLT2)

    import requests
    DLT3=requests.get('http://datachart.500.com/dlt/history/history.shtml')
    DLT4=str(DLT3.text)
    DLT5=re.split(r'<!--<td>2</td>-->|\n',DLT4)
    DLT7=[]
    for i in DLT5:
        if 'cfont' in i or 't_tr1' in i :
            DLT6=re.split(r'<td class="t_tr1">|<td class="cfont2">|<td class="cfont4">|</td>',i)
            if DLT6[0] == '' and len(DLT6[1])==5:
                if DLT6[1] not in DLT2:
                    oknum=''
                    for num in range(1,15):
                        if DLT6[num] != '':
                            oknum = oknum+DLT6[num]+' '
                    oknum = oknum + DLT6[15]
                    #print(oknum)
                    DLT7.append(oknum)

    newDLT=[A[0]]+DLT7+A[1:]
    #for new in newDLT:
        #print(new)
    NEWDLTS = "\n".join(newDLT)
    
    f = open(r'C:/Users/Administrator/Desktop/DLT/DLT2.txt','w',encoding='utf-8')
    f.write(NEWDLTS)
    f.close
    print('数据更新成功！\n执行预测……')
#ONE()
def TWO():
    print('开始预测：'+datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    os_str=r'python c:/Users/Administrator/Desktop/DLT/DLTYC.py'
    f = subprocess.Popen(os_str,shell=True,stdout=subprocess.PIPE)
    #print(f.communicate())
    return str(f.communicate()[0].decode('GB2312'))
    #read().decode("utf-8")
def ding(text):
    from dingtalkchatbot.chatbot import DingtalkChatbot
    webhook = '这里放钉钉的token这里放钉钉的token这里放钉钉的token这里放钉钉的token'
    xiaoding = DingtalkChatbot(webhook)
    #前置信息可以包含在钉钉上设置的关键词，我这里是大家好
    news='大家好！\n以下是今天的体彩预测：\n'+str(text)
    xiaoding.send_text(msg=news, is_at_all=True)
    print('进程结束：'+datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
#定时设置函数
def main(h=10, m=25):
    '''h表示设定的小时，m为设定的分钟'''
    while True:
        # 判断是否达到设定时间，例如23:00
        while True:
            now = datetime.datetime.now()
            # 到达设定时间，结束内循环
            if now.hour==h and now.minute==m and now.isoweekday() in [2,5,7]:
                break
            # 不到时间就等20秒之后再次检测
            time.sleep(20)
        ONE()
        OKSTR = TWO()
        time.sleep(60)
        ding(OKSTR)
if __name__ == '__main__':
    main()





    


    



