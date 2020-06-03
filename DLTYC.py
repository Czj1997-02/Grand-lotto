import numpy as np
import pandas as pd
import math
#pip install -i https://pypi.douban.com/simple math
#基于矩阵计算写的三个时间序列平滑移动线性预测
def Yuce(Y_, T, yece_X_T):  # 对周期为T的数组列表Y_预测未来X个周期的值
    Y = []  # 建立一个空列表用来装格式化后的数据
    for a in Y_:  # 遍历数组中的数据，进行标准格式化
        a_ = float(a)  # 全部浮点化放给a_
        Y.append(a_)  # 把标准化后的数据放进预存列表

    # 1.计算一个周期变化后的连续移动求和:根据数据选取周期，日数据周期为365，周数据周期为52，月数据周期为12
    Last_T_yidong_Sum = []  # 建立一个列表用来存放一个周期之后的移动求和值
    for num_1 in list(range(1, len(Y) + 1 - T)):  # 移动求和值的总数量等于所有数值减去一个周期
        Sum_num = sum(Y[num_1:num_1 + T])  # 设置好求和长度为一个周期T并求和
        Last_T_yidong_Sum.append(Sum_num)  # 把求和结果放入预存列表
    # 2.生成移动求和的预测值
    X = list(range(1, len(Last_T_yidong_Sum) + 1))  # 因为只有一个数组，生成一个与之匹配的标准单位X
    X_pingjun = sum(X) / len(Last_T_yidong_Sum)  # 求单位X的均值
    Y_pingjun = sum(Last_T_yidong_Sum) / len(Last_T_yidong_Sum)  # 求Y项的均值，即前面得出的移动求和值的均值
    Sum_1 = []  # 放Xi*Yi
    Sum_2 = []  # 放Xi*Xi
    for n in list(range(0, len(Last_T_yidong_Sum))):  # 遍历一个和移动求和值数组等位的序数表
        sum_1 = X[n] * Last_T_yidong_Sum[n]  # Xi*Yi
        Sum_1.append(sum_1)  # 对Xi*Yi求合计
        sum_2 = X[n] * X[n]  # Xi*Xi
        Sum_2.append(sum_2)  # 对Xi*Xi求合计
    if (sum(Sum_2) - len(Last_T_yidong_Sum) * X_pingjun * X_pingjun) == 0:
        b = 0  # 剔除被除数为零的情况，被除数为零代表是一条水平线
    else:  # 被除数不为零时，按照回归方程公式计算系数b
        b = (sum(Sum_1) - len(Last_T_yidong_Sum) * X_pingjun * Y_pingjun) / (
                    sum(Sum_2) - len(Last_T_yidong_Sum) * X_pingjun * X_pingjun)
    a = Y_pingjun - b * X_pingjun  # 计算系数a
    Yuce = list(range(len(Last_T_yidong_Sum) + 1, len(Last_T_yidong_Sum) + yece_X_T * T + 1))  # 生成用于预测的X数组
    X_Y_yece_sum = []  # 建立个列表用来放移动合计的预测值
    for yuce_X_num in Yuce:  # 遍历需要预测Y值的X数组
        Y_yuce = b * yuce_X_num + a  # 按照回归系数进行预测
        X_Y_yece_sum.append(Y_yuce)  # 把得到的预测值放入预存列表

    # 3.两个拼接（可能有多余部分，因为是由之前写的外部程序改写的函数，懒得多改）
    every_T = Y[T:]  # 先把原始数据中一个周期时间点后的数据放到每期中
    start_X_T = int(len(Y[T:]))  # 计算原始数据砍去第一个周期后的剩余长度
    for sum_y in X_Y_yece_sum:  # 遍历移动求和的预测值
        every_T.append(sum_y)  # 把移动求和的预测值添加到每期数据后方便做差方便向前做减
    for Y_i in list(range(start_X_T, start_X_T + len(X_Y_yece_sum))):  # 从历史长度之后开始到最后一个位置为止
        every_T[Y_i] = every_T[Y_i] - sum(every_T[Y_i - T + 1:Y_i])  # 减去前面T-1位的合计值（减去前面T-1项的求和）

    # 5.实际得到的X个周期内的各期值
    X_Y_every_yuce = every_T[start_X_T:]  # 把历史数据移除只要预测出来的数据
    return X_Y_every_yuce  # 返回这个预测值列表
def Yuce_DLT(Y_Y, T, yece_X_T):  # 对周期为T的数组列表Y_预测未来X个周期的值
    Y_ = []
    for i in range(1,len(Y_Y)-1):
        e = float(sum(Y_Y[0:i])/len(Y_Y[0:i]))-float(Y_Y[i+1])
        Y_.append(e)
    Y = []  # 建立一个空列表用来装格式化后的数据
    for a in Y_:  # 遍历数组中的数据，进行标准格式化
        a_ = float(a)  # 全部浮点化放给a_
        Y.append(a_)  # 把标准化后的数据放进预存列表

    # 1.计算一个周期变化后的连续移动求和:根据数据选取周期，日数据周期为365，周数据周期为52，月数据周期为12
    Last_T_yidong_Sum = []  # 建立一个列表用来存放一个周期之后的移动求和值
    for num_1 in list(range(1, len(Y) + 1 - T)):  # 移动求和值的总数量等于所有数值减去一个周期
        Sum_num = sum(Y[num_1:num_1 + T])  # 设置好求和长度为一个周期T并求和
        Last_T_yidong_Sum.append(Sum_num)  # 把求和结果放入预存列表
    # 2.生成移动求和的预测值
    X = list(range(1, len(Last_T_yidong_Sum) + 1))  # 因为只有一个数组，生成一个与之匹配的标准单位X
    X_pingjun = sum(X) / len(Last_T_yidong_Sum)  # 求单位X的均值
    Y_pingjun = sum(Last_T_yidong_Sum) / len(Last_T_yidong_Sum)  # 求Y项的均值，即前面得出的移动求和值的均值
    Sum_1 = []  # 放Xi*Yi
    Sum_2 = []  # 放Xi*Xi
    for n in list(range(0, len(Last_T_yidong_Sum))):  # 遍历一个和移动求和值数组等位的序数表
        sum_1 = X[n] * Last_T_yidong_Sum[n]  # Xi*Yi
        Sum_1.append(sum_1)  # 对Xi*Yi求合计
        sum_2 = X[n] * X[n]  # Xi*Xi
        Sum_2.append(sum_2)  # 对Xi*Xi求合计
    if (sum(Sum_2) - len(Last_T_yidong_Sum) * X_pingjun * X_pingjun) == 0:
        b = 0  # 剔除被除数为零的情况，被除数为零代表是一条水平线
    else:  # 被除数不为零时，按照回归方程公式计算系数b
        b = (sum(Sum_1) - len(Last_T_yidong_Sum) * X_pingjun * Y_pingjun) / (sum(Sum_2) - len(Last_T_yidong_Sum) * X_pingjun * X_pingjun)
    a = Y_pingjun - b * X_pingjun  # 计算系数a
    Yuce = list(range(len(Last_T_yidong_Sum) + 1, len(Last_T_yidong_Sum) + yece_X_T * T + 1))  # 生成用于预测的X数组
    X_Y_yece_sum = []  # 建立个列表用来放移动合计的预测值
    for yuce_X_num in Yuce:  # 遍历需要预测Y值的X数组
        Y_yuce = b * yuce_X_num + a  # 按照回归系数进行预测
        X_Y_yece_sum.append(Y_yuce)  # 把得到的预测值放入预存列表

    # 3.两个拼接（可能有多余部分，因为是由之前写的外部程序改写的函数，懒得多改）
    every_T = Y[T:]  # 先把原始数据中一个周期时间点后的数据放到每期中
    start_X_T = int(len(Y[T:]))  # 计算原始数据砍去第一个周期后的剩余长度
    for sum_y in X_Y_yece_sum:  # 遍历移动求和的预测值
        every_T.append(sum_y)  # 把移动求和的预测值添加到每期数据后方便做差方便向前做减
    for Y_i in list(range(start_X_T, start_X_T + len(X_Y_yece_sum))):  # 从历史长度之后开始到最后一个位置为止
        every_T[Y_i] = every_T[Y_i] - sum(every_T[Y_i - T + 1:Y_i])  # 减去前面T-1位的合计值（减去前面T-1项的求和）

    # 5.实际得到的X个周期内的各期值
    X_Y_every_yuce = every_T[start_X_T:]  # 把历史数据移除只要预测出来的数据
    return float(sum(Y_Y)/len(Y_Y))-float(X_Y_every_yuce[0])  # 返回这个预测值列表
def Yuce_DLT_e(Y_Y, T, yece_X_T):  # 对周期为T的数组列表Y_预测未来X个周期的值
    Y_all = []
    for i in range(1,len(Y_Y)-1):
        e = float(Y_Y[i+1])/float(sum(Y_Y[0:i])/len(Y_Y[0:i]))
        Y_all.append(e)

    Y_max=[]
    Y_min=[]
    for i in Y_all:
        if int(i)>=1:
            Y_max.append(i)
        elif int(i)<1:
            Y_min.append(i)

    Y_ = []
    if len(Y_max)>=len(Y_min):
        for k in Y_min:
            Y_.append(k)
    else:
        for k in Y_max:
            Y_.append(k)
    Y = []  # 建立一个空列表用来装格式化后的数据
    for a in Y_:  # 遍历数组中的数据，进行标准格式化
        a_ = float(a)  # 全部浮点化放给a_
        Y.append(a_)  # 把标准化后的数据放进预存列表

    # 1.计算一个周期变化后的连续移动求和:根据数据选取周期，日数据周期为365，周数据周期为52，月数据周期为12
    Last_T_yidong_Sum = []  # 建立一个列表用来存放一个周期之后的移动求和值
    for num_1 in list(range(1, len(Y) + 1 - T)):  # 移动求和值的总数量等于所有数值减去一个周期
        Sum_num = sum(Y[num_1:num_1 + T])  # 设置好求和长度为一个周期T并求和
        Last_T_yidong_Sum.append(Sum_num)  # 把求和结果放入预存列表
    # 2.生成移动求和的预测值
    X = list(range(1, len(Last_T_yidong_Sum) + 1))  # 因为只有一个数组，生成一个与之匹配的标准单位X
    X_pingjun = sum(X) / len(Last_T_yidong_Sum)  # 求单位X的均值
    Y_pingjun = sum(Last_T_yidong_Sum) / len(Last_T_yidong_Sum)  # 求Y项的均值，即前面得出的移动求和值的均值
    Sum_1 = []  # 放Xi*Yi
    Sum_2 = []  # 放Xi*Xi
    for n in list(range(0, len(Last_T_yidong_Sum))):  # 遍历一个和移动求和值数组等位的序数表
        sum_1 = X[n] * Last_T_yidong_Sum[n]  # Xi*Yi
        Sum_1.append(sum_1)  # 对Xi*Yi求合计
        sum_2 = X[n] * X[n]  # Xi*Xi
        Sum_2.append(sum_2)  # 对Xi*Xi求合计
    if (sum(Sum_2) - len(Last_T_yidong_Sum) * X_pingjun * X_pingjun) == 0:
        b = 0  # 剔除被除数为零的情况，被除数为零代表是一条水平线
    else:  # 被除数不为零时，按照回归方程公式计算系数b
        b = (sum(Sum_1) - len(Last_T_yidong_Sum) * X_pingjun * Y_pingjun) / (sum(Sum_2) - len(Last_T_yidong_Sum) * X_pingjun * X_pingjun)
    a = Y_pingjun - b * X_pingjun  # 计算系数a
    Yuce = list(range(len(Last_T_yidong_Sum) + 1, len(Last_T_yidong_Sum) + yece_X_T * T + 1))  # 生成用于预测的X数组
    X_Y_yece_sum = []  # 建立个列表用来放移动合计的预测值
    for yuce_X_num in Yuce:  # 遍历需要预测Y值的X数组
        Y_yuce = b * yuce_X_num + a  # 按照回归系数进行预测
        X_Y_yece_sum.append(Y_yuce)  # 把得到的预测值放入预存列表

    # 3.两个拼接（可能有多余部分，因为是由之前写的外部程序改写的函数，懒得多改）
    every_T = Y[T:]  # 先把原始数据中一个周期时间点后的数据放到每期中
    start_X_T = int(len(Y[T:]))  # 计算原始数据砍去第一个周期后的剩余长度
    for sum_y in X_Y_yece_sum:  # 遍历移动求和的预测值
        every_T.append(sum_y)  # 把移动求和的预测值添加到每期数据后方便做差方便向前做减
    for Y_i in list(range(start_X_T, start_X_T + len(X_Y_yece_sum))):  # 从历史长度之后开始到最后一个位置为止
        every_T[Y_i] = every_T[Y_i] - sum(every_T[Y_i - T + 1:Y_i])  # 减去前面T-1位的合计值（减去前面T-1项的求和）

    # 5.实际得到的X个周期内的各期值
    X_Y_every_yuce = every_T[start_X_T:]  # 把历史数据移除只要预测出来的数据
    return float(sum(Y_Y)/len(Y_Y))*float(X_Y_every_yuce[0])  # 返回这个预测值列表

#0-1.导入数据

R=r'C:/Users/Administrator/Desktop/DLT/DLT2.txt'                                  #路径绝对化
data = pd.read_csv(R, sep=' ', header=None)     #读取路径文件数据
data = data.sort_index(ascending=False).values  #数据反过来
data = data[:, 1:]                              #按照格式分隔数据
time_=len(data)-2                               #计算次序
#0-2.数据列表化（便于使用矩阵计算）
no1 = []
no2 = []
no3 = []
no4 = []
no5 = []
blue1 = []
blue2 = []
for i in range(0,len(data)):
    no1.append(data[i][0])
    no2.append(data[i][1])
    no3.append(data[i][2])
    no4.append(data[i][3])
    no5.append(data[i][4])
    blue1.append(data[i][5])
    blue2.append(data[i][6])


#1.机器训练(概率变动)预测
def fengbu(i):
    abb = {}
    for l in range(7):
        for n in range(1, 36):
            abb[l, n] = []
            for qiu in range(i - 1):
                if data[qiu][l] == n:
                    a = data[qiu + 1][l] - data[qiu][l]
                    abb[l, n].append(a)  # 一个大字典为{（l,n):a}
    dict1 = {}
    dict2 = {}  # 每个数字增大的概率
    add1 = {}  # 增大的次数
    reduce = {}  # 减小的次数
    da = {}
    jian = {}
    da1 = []
    jian1 = []
    dict21 = []
    for n, l in abb.items():
        add1[n] = 0
        reduce[n] = 0
        da[n] = 0
        jian[n] = 0
        for m in l:
            if m > 0:
                add1[n] += 1  # 统计往期为这个数字时下次增大次数
            elif m < 0:
                reduce[n] += 1  # 减小次数

        dict2[n] = round(add1[n] / (reduce[n] + add1[n] + 1), 4)
        # 得到前面那张概率图 减小和它相反
        for m in set(l):
            if m > 0:
                dict1[n, m] = (round(l.count(m) / add1[n], 4)) * m
                da[n] += dict1[n, m]
                '''
                这是基于首先判断当前期每个数字增大或减小概率哪个大
                数值大的进一步细化，即将具体增大或减小的值得概率当
                成权重再分别与之对应值相乘,在全部相加为下一次预测值

                '''
            elif m < 0:
                dict1[n, m] = (round(l.count(m) / reduce[n], 4)) * m
                jian[n] += dict1[n, m]
            elif m == 0:
                dict1[n, m] = 0  # 两次数字不变
    for n, m, l in zip(da.values(), jian.values(), dict2.values()):
        da1.append(n)  # 原来是字典现在要将其弄成矩阵
        jian1.append(m)
        dict21.append(l)
    da1 = np.array(da1).reshape(7, 35)
    jian1 = np.array(jian1).reshape(7, 35)
    dict21 = np.array(dict21).reshape(7, 35)
    # shuan
    return da1, jian1, dict21
def predict(i):
    #for red in range(7):
        #print(round(data[:, red].mean(), 4), round(data[:, red].std(), 4))
        #当前均值
        #方差
    da1, jian1, dict21 = fengbu(i)
    predict = np.zeros(7)
    predictmin = np.zeros(7)
    predictmax = np.zeros(7)
    predictsuper = np.zeros(7)
    for l in range(7):
        for m in range(1, 36):
            if data[i][l] == m:
                if dict21[l][m - 1] > 0.5:
                    #print(dict21[l][m - 1], da1[l][m - 1], data[i][l])
                    # 每期每个数字增大或减小概率，权重和，每个数字值
                    predict[l] = data[i][l] + da1[l][m - 1]
                    predictmin[l] = int(data[i][l] + da1[l][m - 1])
                    predictmax[l] = round(data[i][l] + da1[l][m - 1])
                    predictsuper[l] = math.ceil(data[i][l] + da1[l][m - 1])
                elif dict21[l][m - 1] < 0.5:
                    #print(dict21[l][m - 1], jian1[l][m - 1], data[i][l])
                    predict[l] = data[i][l] + jian1[l][m - 1]
                    predictmin[l] = int(data[i][l] + jian1[l][m - 1])
                    predictmax[l] = round(data[i][l] + jian1[l][m - 1])
                    predictsuper[l] = math.ceil(data[i][l] + jian1[l][m - 1])

    OUT1="上一次 \n第 %d次,结果是:\n%s" % (time_, data[time_])+'\n'+'-' * 25
    OUT2='机器训练--概率变动预测：\n'+"%s" % predict+"\n%s" % predictmin+"\n%s" % predictmax+"\n%s" % predictsuper+'\n'+'-'*25
    print(OUT1)
    print(OUT2)
#2.移动平滑曲线预测
def shuxue():
    #滚动记差
    i1=[]
    for i in range(2,len(no1)-1):
        q = no1[0:i]
        Y = Yuce(q,1,1)
        e = no1[i+1] - Y[0]
        i1.append(e)
    q1=Yuce(no1,1,1)
    i1e=Yuce(i1,1,1)
    i2=[]
    for i in range(2,len(no2)-1):
        q = no2[0:i]
        Y = Yuce(q,1,1)
        e = no2[i+1] - Y[0]
        i2.append(e)
    q2=Yuce(no2,1,1)
    i2e=Yuce(i2,1,1)
    i3=[]
    for i in range(2,len(no3)-1):
        q = no3[0:i]
        Y = Yuce(q,1,1)
        e = no3[i+1] - Y[0]
        i3.append(e)
    q3=Yuce(no3,1,1)
    i3e=Yuce(i3,1,1)
    i4=[]
    for i in range(2,len(no4)-1):
        q = no4[0:i]
        Y = Yuce(q,1,1)
        e = no4[i+1] - Y[0]
        i4.append(e)
    q4=Yuce(no4,1,1)
    i4e=Yuce(i4,1,1)
    i5=[]
    for i in range(2,len(no5)-1):
        q = no5[0:i]
        Y = Yuce(q,1,1)
        e = no5[i+1] - Y[0]
        i5.append(e)
    q5=Yuce(no5,1,1)
    i5e=Yuce(i5,1,1)
    i6=[]
    for i in range(2,len(blue1)-1):
        q = blue1[0:i]
        Y = Yuce(q,1,1)
        e = blue1[i+1] - Y[0]
        i6.append(e)
    q6=Yuce(blue1,1,1)
    i6e=Yuce(i6,1,1)
    i7=[]
    for i in range(2,len(blue2)-1):
        q = blue2[0:i]
        Y = Yuce(q,1,1)
        e = blue2[i+1] - Y[0]
        i7.append(e)
    q7=Yuce(blue2,1,1)
    i7e=Yuce(i7,1,1)

    QQ1=int((q1[0]+i1e[0])*10000)/10000
    QQ2=int((q2[0]+i2e[0])*10000)/10000
    QQ3=int((q3[0]+i3e[0])*10000)/10000
    QQ4=int((q4[0]+i4e[0])*10000)/10000
    QQ5=int((q5[0]+i5e[0])*10000)/10000
    QQ6=int((q6[0]+i6e[0])*10000)/10000
    QQ7=int((q7[0]+i7e[0])*10000)/10000
    print('平偏振滑--滚动记差偏振：')
    #print('[ '+str(QQ1)+' '+str(QQ2)+' '+str(QQ3)+' '+str(QQ4)+' '+str(QQ5)+'  '+str(QQ6)+' '+str(QQ7)+' ]')
    print('[ '+str(int(QQ1)-1)+'. '+str(int(QQ2)-1)+'. '+str(int(QQ3)-1)+'. '+str(int(QQ4)-1)+'. '+str(int(QQ5)-1)+'.  '+str(int(QQ6)-1)+'. '+str(int(QQ7)-1)+'.]')
    print('[ '+str(round(QQ1))+'. '+str(round(QQ2))+'. '+str(round(QQ3))+'. '+str(round(QQ4))+'. '+str(round(QQ5))+'.  '+str(round(QQ6))+'. '+str(round(QQ7))+'.]')
    print('[ '+str(math.ceil(QQ1)+1)+'. '+str(math.ceil(QQ2)+1)+'. '+str(math.ceil(QQ3)+1)+'. '+str(math.ceil(QQ4)+1)+'. '+str(math.ceil(QQ5)+1)+'.  '+str(math.ceil(QQ6)+1)+'. '+str(math.ceil(QQ7)+1)+'.]')
    print('-' * 25)
    print('移动平滑--滚动记差预测：')
    print('[ '+str(QQ1)+' '+str(QQ2)+' '+str(QQ3)+' '+str(QQ4)+' '+str(QQ5)+'  '+str(QQ6)+' '+str(QQ7)+' ]')
    print('[ '+str(int(QQ1))+'. '+str(int(QQ2))+'. '+str(int(QQ3))+'. '+str(int(QQ4))+'. '+str(int(QQ5))+'.  '+str(int(QQ6))+'. '+str(int(QQ7))+'.]')
    print('[ '+str(round(QQ1))+'. '+str(round(QQ2))+'. '+str(round(QQ3))+'. '+str(round(QQ4))+'. '+str(round(QQ5))+'.  '+str(round(QQ6))+'. '+str(round(QQ7))+'.]')
    print('[ '+str(math.ceil(QQ1))+'. '+str(math.ceil(QQ2))+'. '+str(math.ceil(QQ3))+'. '+str(math.ceil(QQ4))+'. '+str(math.ceil(QQ5))+'.  '+str(math.ceil(QQ6))+'. '+str(math.ceil(QQ7))+'.]')
    print('-' * 25)
    OUT3 = '移动平滑--滚动记差预测：'+'\n'+'所以预测下一次是:'+'[ '+str(QQ1)+' '+str(QQ2)+' '+str(QQ3)+' '+str(QQ4)+' '+str(QQ5)+'  '+str(QQ6)+' '+str(QQ7)+' ]'+'\n'+'预测值向下舍入是:'+'[ '+str(int(QQ1))+'. '+str(int(QQ2))+'. '+str(int(QQ3))+'. '+str(int(QQ4))+'. '+str(int(QQ5))+'.  '+str(int(QQ6))+'. '+str(int(QQ7))+'.]'+'\n'+'预测值四舍五入是:'+'[ '+str(round(QQ1))+'. '+str(round(QQ2))+'. '+str(round(QQ3))+'. '+str(round(QQ4))+'. '+str(round(QQ5))+'.  '+str(round(QQ6))+'. '+str(round(QQ7))+'.]'+'\n'+'预测值向上舍入是:'+'[ '+str(math.ceil(QQ1))+'. '+str(math.ceil(QQ2))+'. '+str(math.ceil(QQ3))+'. '+str(math.ceil(QQ4))+'. '+str(math.ceil(QQ5))+'.  '+str(math.ceil(QQ6))+'. '+str(math.ceil(QQ7))+'.]'+'\n'+'-' * 25
    #均值差异
    qq1=int(Yuce_DLT(no1[0:len(no1)-2],1,1)*10000)/10000
    qq2=int(Yuce_DLT(no2[0:len(no2)-2],1,1)*10000)/10000
    qq3=int(Yuce_DLT(no3[0:len(no3)-2],1,1)*10000)/10000
    qq4=int(Yuce_DLT(no4[0:len(no4)-2],1,1)*10000)/10000
    qq5=int(Yuce_DLT(no5[0:len(no5)-2],1,1)*10000)/10000
    qq6=int(Yuce_DLT(blue1[0:len(blue1)-2],1,1)*10000)/10000
    qq7=int(Yuce_DLT(blue2[0:len(blue2)-2],1,1)*10000)/10000
    print('移动平滑--平均差预测：')
    print('[ '+str(qq1)+' '+str(qq2)+' '+str(qq3)+' '+str(qq4)+' '+str(qq5)+'  '+str(qq6)+' '+str(qq7)+' ]')
    print('[ '+str(int(qq1))+'. '+str(int(qq2))+'. '+str(int(qq3))+'. '+str(int(qq4))+'. '+str(int(qq5))+'.  '+str(int(qq6))+'. '+str(int(qq7))+'.]')
    print('[ '+str(round(qq1))+'. '+str(round(qq2))+'. '+str(round(qq3))+'. '+str(round(qq4))+'. '+str(round(qq5))+'.  '+str(round(qq6))+'. '+str(round(qq7))+'.]')
    print('[ '+str(math.ceil(qq1))+'. '+str(math.ceil(qq2))+'. '+str(math.ceil(qq3))+'. '+str(math.ceil(qq4))+'. '+str(math.ceil(qq5))+'.  '+str(math.ceil(qq6))+'. '+str(math.ceil(qq7))+'.]')
    print('-' * 25)
    OUT4=('移动平滑--平均差预测：')+'\n'+('所以预测下一次是:'+'[ '+str(qq1)+' '+str(qq2)+' '+str(qq3)+' '+str(qq4)+' '+str(qq5)+'  '+str(qq6)+' '+str(qq7)+' ]')+'\n'+('预测值向下舍入是:'+'[ '+str(int(qq1))+'. '+str(int(qq2))+'. '+str(int(qq3))+'. '+str(int(qq4))+'. '+str(int(qq5))+'.  '+str(int(qq6))+'. '+str(int(qq7))+'.]')+'\n'+('预测值四舍五入是:'+'[ '+str(round(qq1))+'. '+str(round(qq2))+'. '+str(round(qq3))+'. '+str(round(qq4))+'. '+str(round(qq5))+'.  '+str(round(qq6))+'. '+str(round(qq7))+'.]')+'\n'+('预测值向上舍入是:'+'[ '+str(math.ceil(qq1))+'. '+str(math.ceil(qq2))+'. '+str(math.ceil(qq3))+'. '+str(math.ceil(qq4))+'. '+str(math.ceil(qq5))+'.  '+str(math.ceil(qq6))+'. '+str(math.ceil(qq7))+'.]')+'\n'+('-' * 75)
if __name__ == '__main__':
    predict(time_)
    shuxue()

#其他方法（指数平滑
e1box=[]
for i in range(149,len(no1)-1):
    ee = float(Yuce_DLT(no1[0:i],1,1))-float(no1[i+1])
    e1box.append(ee)
qq01=int((Yuce_DLT(no1[149:len(no1)-2],1,1)-Yuce(e1box,1,1)[0])*10000)/10000

if float(qq01)<0:
    qq01 = float(qq01) + math.ceil(abs(float(qq01))/35)*35

e2box=[]
for i in range(149,len(no2)-1):
    ee = float(Yuce_DLT(no2[0:i],1,1))-float(no2[i+1])
    e2box.append(ee)
qq02=int((Yuce_DLT(no2[149:len(no2)-2],1,1)-Yuce(e2box,1,1)[0])*10000)/10000

if float(qq02)<0:
    qq02 = float(qq02) + math.ceil(abs(float(qq02))/35)*35

        
e3box=[]
for i in range(149,len(no3)-1):
    ee = float(Yuce_DLT(no3[0:i],1,1))-float(no3[i+1])
    e3box.append(ee)
qq03=int((Yuce_DLT(no3[149:len(no3)-2],1,1)-Yuce(e3box,1,1)[0])*10000)/10000

if float(qq03)<0:
    qq03 = float(qq03) + math.ceil(abs(float(qq03))/35)*35

        
e4box=[]
for i in range(149,len(no4)-1):
    ee = float(Yuce_DLT(no4[0:i],1,1))-float(no4[i+1])
    e4box.append(ee)
qq04=int((Yuce_DLT(no4[149:len(no4)-2],1,1)-Yuce(e4box,1,1)[0])*10000)/10000

if float(qq04)<0:
    qq04 = float(qq04) + math.ceil(abs(float(qq04))/35)*35

        
e5box=[]
for i in range(149,len(no5)-1):
    ee = float(Yuce_DLT(no5[0:i],1,1))-float(no5[i+1])
    e5box.append(ee)
qq05=int((Yuce_DLT(no5[149:len(no5)-2],1,1)-Yuce(e5box,1,1)[0])*10000)/10000

if float(qq05)<0:
    qq05 = float(qq05) + math.ceil(abs(float(qq05))/35)*35

        
e6box=[]
for i in range(149,len(blue1)-1):
    ee = float(Yuce_DLT(blue1[0:i],1,1))-float(blue1[i+1])
    e6box.append(ee)
qq06=int((Yuce_DLT(blue1[149:len(blue1)-2],1,1)-Yuce(e6box,1,1)[0])*10000)/10000

if float(qq06)<0:
    qq06 = float(qq06) + math.ceil(abs(float(qq06))/12)*12

        
e7box=[]
for i in range(149,len(blue2)-1):
    ee = float(Yuce_DLT(blue2[0:i],1,1))-float(blue2[i+1])
    e7box.append(ee)
qq07=int((Yuce_DLT(blue2[149:len(blue2)-2],1,1)-Yuce(e7box,1,1)[0])*10000)/10000

if float(qq07)<0:
    qq07 = float(qq07) + math.ceil(abs(float(qq07))/12)*12

print('其他--二次平均差预测：')
print('[ '+str(qq01)+' '+str(qq02)+' '+str(qq03)+' '+str(qq04)+' '+str(qq05)+'  '+str(qq06)+' '+str(qq07)+' ]')
print('[ '+str(int(qq01))+'. '+str(int(qq02))+'. '+str(int(qq03))+'. '+str(int(qq04))+'. '+str(int(qq05))+'.  '+str(int(qq06))+'. '+str(int(qq07))+'.]')
print('[ '+str(round(qq01))+'. '+str(round(qq02))+'. '+str(round(qq03))+'. '+str(round(qq04))+'. '+str(round(qq05))+'.  '+str(round(qq06))+'. '+str(round(qq07))+'.]')
print('[ '+str(math.ceil(qq01))+'. '+str(math.ceil(qq02))+'. '+str(math.ceil(qq03))+'. '+str(math.ceil(qq04))+'. '+str(math.ceil(qq05))+'.  '+str(math.ceil(qq06))+'. '+str(math.ceil(qq07))+'.]')
print('-' * 25)

#1.计算一定样本量下预测值和真实值的差异
k1=[]
for i in range(149,len(no1)-1):
    k = Yuce_DLT_e(no1[0:i],1,1)-float(no1[i+1])
    k1.append(k)
#2.Yn+1=Y预测-k预测
Y1 = int(Yuce_DLT_e(no1[149:len(no1)-2],1,1)-float(Yuce(k1,1,1)[0])*10000)/10000

if float(Y1)<0:
    Y1 = float(Y1) + math.ceil(abs(float(Y1))/35)*35

#1.计算一定样本量下预测值和真实值的差异
k2=[]
for i in range(149,len(no2)-1):
    k = Yuce_DLT_e(no2[0:i],1,1)-float(no2[i+1])
    k2.append(k)
#2.Yn+1=Y预测-k预测
Y2 =  int(Yuce_DLT_e(no2[149:len(no2)-2],1,1)-float(Yuce(k2,1,1)[0])*10000)/10000

if float(Y2)<0:
    Y2 = float(Y2) + math.ceil(abs(float(Y2))/35)*35
        
#1.计算一定样本量下预测值和真实值的差异
k3=[]
for i in range(149,len(no3)-1):
    k = Yuce_DLT_e(no3[0:i],1,1)-float(no3[i+1])
    k3.append(k)
#2.Yn+1=Y预测-k预测
Y3 =  int(Yuce_DLT_e(no3[149:len(no3)-2],1,1)-float(Yuce(k3,1,1)[0])*10000)/10000

if float(Y3)<0:
    Y3 = float(Y3) + math.ceil(abs(float(Y3))/35)*35
        
#1.计算一定样本量下预测值和真实值的差异
k4=[]
for i in range(149,len(no4)-1):
    k = Yuce_DLT_e(no4[0:i],1,1)-float(no4[i+1])
    k4.append(k)
#2.Yn+1=Y预测-k预测
Y4 =  int(Yuce_DLT_e(no4[149:len(no4)-2],1,1)-float(Yuce(k4,1,1)[0])*10000)/10000

if float(Y4)<0:
    Y4 = float(Y4) + math.ceil(abs(float(Y4))/35)*35
        
#1.计算一定样本量下预测值和真实值的差异
k5=[]
for i in range(149,len(no5)-1):
    k = Yuce_DLT_e(no5[0:i],1,1)-float(no5[i+1])
    k5.append(k)
#2.Yn+1=Y预测-k预测
Y5 =  int(Yuce_DLT_e(no5[149:len(no5)-2],1,1)-float(Yuce(k5,1,1)[0])*10000)/10000

if float(Y5)<0:
    Y5 = float(Y5) + math.ceil(abs(float(Y5))/35)*35
        
#1.计算一定样本量下预测值和真实值的差异
k6=[]
for i in range(149,len(blue1)-1):
    k = Yuce_DLT_e(blue1[0:i],1,1)-float(blue1[i+1])
    k6.append(k)
#2.Yn+1=Y预测-k预测
Y6 =  int(Yuce_DLT_e(blue1[149:len(blue1)-2],1,1)-float(Yuce(k6,1,1)[0])*10000)/10000

if float(Y6)<0:
    Y6 = float(Y6) + math.ceil(abs(float(Y6))/12)*12
        
#1.计算一定样本量下预测值和真实值的差异
k7=[]
for i in range(149,len(blue2)-1):
    k = Yuce_DLT_e(blue2[0:i],1,1)-float(blue2[i+1])
    k7.append(k)
#2.Yn+1=Y预测-k预测
Y7 =  int(Yuce_DLT_e(blue2[149:len(blue2)-2],1,1)-float(Yuce(k7,1,1)[0])*10000)/10000

if float(Y7)<0:
    Y7 = float(Y7) + math.ceil(abs(float(Y7))/12)*12
print('多元回归偏振预测：')
#print('[ '+str(Y1)+' '+str(Y2)+' '+str(Y3)+' '+str(Y4)+' '+str(Y5)+'  '+str(Y6)+' '+str(Y7)+' ]')
print('[ '+str(int(Y1)-1)+'. '+str(int(Y2)-1)+'. '+str(int(Y3)-1)+'. '+str(int(Y4)-1)+'. '+str(int(Y5)-1)+'.  '+str(int(Y6)-1)+'. '+str(int(Y7)-1)+'.]')
print('[ '+str(round(Y1))+'. '+str(round(Y2))+'. '+str(round(Y3))+'. '+str(round(Y4))+'. '+str(round(Y5))+'.  '+str(round(Y6))+'. '+str(round(Y7))+'.]')
print('[ '+str(math.ceil(Y1)+1)+'. '+str(math.ceil(Y2)+1)+'. '+str(math.ceil(Y3)+1)+'. '+str(math.ceil(Y4)+1)+'. '+str(math.ceil(Y5)+1)+'.  '+str(math.ceil(Y6)+1)+'. '+str(math.ceil(Y7)+1)+'.]')
print('-' * 25)
print('其他--多元回归预测：')
print('[ '+str(Y1)+' '+str(Y2)+' '+str(Y3)+' '+str(Y4)+' '+str(Y5)+'  '+str(Y6)+' '+str(Y7)+' ]')
print('[ '+str(int(Y1))+'. '+str(int(Y2))+'. '+str(int(Y3))+'. '+str(int(Y4))+'. '+str(int(Y5))+'.  '+str(int(Y6))+'. '+str(int(Y7))+'.]')
print('[ '+str(round(Y1))+'. '+str(round(Y2))+'. '+str(round(Y3))+'. '+str(round(Y4))+'. '+str(round(Y5))+'.  '+str(round(Y6))+'. '+str(round(Y7))+'.]')
print('[ '+str(math.ceil(Y1))+'. '+str(math.ceil(Y2))+'. '+str(math.ceil(Y3))+'. '+str(math.ceil(Y4))+'. '+str(math.ceil(Y5))+'.  '+str(math.ceil(Y6))+'. '+str(math.ceil(Y7))+'.]')
print('-' * 25)

#指数平滑算法
def exponential_smoothing(alpha, s):
    '''
    一次指数平滑
    :param alpha:  平滑系数
    :param s:      数据序列， list
    :return:       返回一次指数平滑模型参数， list
    '''
    s_temp = [0 for i in range(len(s))]
    s_temp[0] = ( s[0] + s[1] + s[2] ) / 3
    for i in range(1, len(s)):
        s_temp[i] = alpha * s[i] + (1 - alpha) * s_temp[i-1]
    return s_temp
def compute_single(alpha, s):
    '''
    一次指数平滑
    :param alpha:  平滑系数
    :param s:      数据序列， list
    :return:       返回一次指数平滑模型参数， list
    '''
    return exponential_smoothing(alpha, s)
def compute_double(alpha, s):
    '''
    二次指数平滑
    :param alpha:  平滑系数
    :param s:      数据序列， list
    :return:       返回二次指数平滑模型参数a, b， list
    '''
    s_single = compute_single(alpha, s)
    s_double = compute_single(alpha, s_single)
    a_double = [0 for i in range(len(s))]
    b_double = [0 for i in range(len(s))]
    for i in range(len(s)):
        a_double[i] = 2 * s_single[i] - s_double[i]                    #计算二次指数平滑的a
        b_double[i] = (alpha / (1 - alpha)) * (s_single[i] - s_double[i])  #计算二次指数平滑的b

    return a_double, b_double
def compute_triple(alpha, s):
    '''
    三次指数平滑
    :param alpha:  平滑系数
    :param s:      数据序列， list
    :return:       返回三次指数平滑模型参数a, b, c， list
    '''
    s_single = compute_single(alpha, s)
    s_double = compute_single(alpha, s_single)
    s_triple = exponential_smoothing(alpha, s_double)
    a_triple = [0 for i in range(len(s))]
    b_triple = [0 for i in range(len(s))]
    c_triple = [0 for i in range(len(s))]
    for i in range(len(s)):
        a_triple[i] = 3 * s_single[i] - 3 * s_double[i] + s_triple[i]
        b_triple[i] = (alpha / (2 * ((1 - alpha) ** 2))) * ((6 - 5 * alpha) * s_single[i] - 2 * ((5 - 4 * alpha) * s_double[i]) + (4 - 3 * alpha) * s_triple[i])
        c_triple[i] = ((alpha ** 2) / (2 * ((1 - alpha) ** 2))) * (s_single[i] - 2 * s_double[i] + s_triple[i])
    return a_triple[-1]+b_triple[-1]+c_triple[-1]
if __name__ == "__main__":
    alpha = float(0.618)

    DE1 = []
    for i in range(149,len(no1)-1):
        D = no1[0:i]
        E = compute_triple(alpha, D)-no1[i+1]
        DE1.append(E)
    data1 = no1[0:len(no1)-2]
    tr1 = int(compute_triple(alpha, data1)*10000)/10000

    if float(tr1)<0:
        tr1 = float(tr1) + math.ceil(abs(float(tr1))/35)*35

    tr_1 = int((compute_triple(alpha, data1)-compute_triple(alpha, DE1))*10000)/10000
    if float(tr_1)<0:
        tr_1 = float(tr_1) + math.ceil(abs(float(tr_1))/35)*35

    DE2 = []
    for i in range(149,len(no2)-1):
        D = no2[0:i]
        E = compute_triple(alpha, D)-no2[i+1]
        DE2.append(E)

    data2 = no2[0:len(no2)-2]
    tr2 = int(compute_triple(alpha, data2)*10000)/10000
    if float(tr2)<0:
        tr2 = float(tr2) + math.ceil(abs(float(tr2))/35)*35
    tr_2 = int((compute_triple(alpha, data2)-compute_triple(alpha, DE2))*10000)/10000
    if float(tr_2)<0:
        tr_2 = float(tr_2) + math.ceil(abs(float(tr_2))/35)*35

    DE3 = []
    for i in range(149,len(no3)-1):
        D = no3[0:i]
        E = compute_triple(alpha, D)-no3[i+1]
        DE3.append(E)

    data3 = no3[0:len(no3)-2]
    tr3 = int(compute_triple(alpha, data3)*10000)/10000
    if float(tr3)<0:
        tr3 = float(tr3) + math.ceil(abs(float(tr3))/35)*35
    tr_3 = int((compute_triple(alpha, data3)-compute_triple(alpha, DE3))*10000)/10000
    if float(tr_3)<0:
        tr_3 = float(tr_3) + math.ceil(abs(float(tr_3))/35)*35

    DE4 = []
    for i in range(149,len(no4)-1):
        D = no4[0:i]
        E = compute_triple(alpha, D)-no4[i+1]
        DE4.append(E)

    data4 = no4[0:len(no4)-2]
    tr4 = int(compute_triple(alpha, data4)*10000)/10000
    if float(tr4)<0:
        tr4 = float(tr4) + math.ceil(abs(float(tr4))/35)*35
    tr_4 = int((compute_triple(alpha, data4)-compute_triple(alpha, DE4))*10000)/10000
    if float(tr_4)<0:
        tr_4 = float(tr_4) + math.ceil(abs(float(tr_4))/35)*35

    DE5 = []
    for i in range(149,len(no5)-1):
        D = no5[0:i]
        E = compute_triple(alpha, D)-no5[i+1]
        DE5.append(E)

    data5 = no5[0:len(no5)-2]
    tr5 = int(compute_triple(alpha, data5)*10000)/10000
    if float(tr5)<0:
        tr5 = float(tr5) + math.ceil(abs(float(tr5))/35)*35
    tr_5 = int((compute_triple(alpha, data5)-compute_triple(alpha, DE5))*10000)/10000
    if float(tr_5)<0:
        tr_5 = float(tr_5) + math.ceil(abs(float(tr_5))/35)*35

    DE6 = []
    for i in range(149,len(blue1)-1):
        D = blue1[0:i]
        E = compute_triple(alpha, D)-blue1[i+1]
        DE6.append(E)

    data6 = blue1[0:len(blue1)-2]
    tr6 = int(compute_triple(alpha, data6)*10000)/10000
    if float(tr6)<0:
        tr6 = float(tr6) + math.ceil(abs(float(tr6))/12)*12
    tr_6 = int((compute_triple(alpha, data6)-compute_triple(alpha, DE6))*10000)/10000
    if float(tr_6)<0:
        tr_6 = float(tr_6) + math.ceil(abs(float(tr_6))/12)*12

    DE7 = []
    for i in range(149,len(blue2)-1):
        D = blue2[0:i]
        E = compute_triple(alpha, D)-blue2[i+1]
        DE7.append(E)

    data7 = blue2[0:len(blue2)-2]
    tr7 = int(compute_triple(alpha, data7)*10000)/10000
    if float(tr7)<0:
        tr7 = float(tr7) + math.ceil(abs(float(tr7))/12)*12
    tr_7 = int((compute_triple(alpha, data7)-compute_triple(alpha, DE7))*10000)/10000
    if float(tr_7)<0:
        tr_7 = float(tr_7) + math.ceil(abs(float(tr_7))/12)*12

print('其他--三次指数平滑预测：')
print('[ '+str(tr1)+' '+str(tr2)+' '+str(tr3)+' '+str(tr4)+' '+str(tr5)+'  '+str(tr6)+' '+str(tr7)+' ]')
print('[ '+str(int(tr1))+'. '+str(int(tr2))+'. '+str(int(tr3))+'. '+str(int(tr4))+'. '+str(int(tr5))+'.  '+str(int(tr6))+'. '+str(int(tr7))+'.]')
print('[ '+str(round(tr1))+'. '+str(round(tr2))+'. '+str(round(tr3))+'. '+str(round(tr4))+'. '+str(round(tr5))+'.  '+str(round(tr6))+'. '+str(round(tr7))+'.]')
print('[ '+str(math.ceil(tr1))+'. '+str(math.ceil(tr2))+'. '+str(math.ceil(tr3))+'. '+str(math.ceil(tr4))+'. '+str(math.ceil(tr5))+'.  '+str(math.ceil(tr6))+'. '+str(math.ceil(tr7))+'.]')
print('-' * 25)
print('其他--三次残差平滑预测：')
print('[ '+str(tr_1)+' '+str(tr_2)+' '+str(tr_3)+' '+str(tr_4)+' '+str(tr_5)+'  '+str(tr_6)+' '+str(tr_7)+' ]')
print('[ '+str(int(tr_1))+'. '+str(int(tr_2))+'. '+str(int(tr_3))+'. '+str(int(tr_4))+'. '+str(int(tr_5))+'.  '+str(int(tr_6))+'. '+str(int(tr_7))+'.]')
print('[ '+str(round(tr_1))+'. '+str(round(tr_2))+'. '+str(round(tr_3))+'. '+str(round(tr_4))+'. '+str(round(tr_5))+'.  '+str(round(tr_6))+'. '+str(round(tr_7))+'.]')
print('[ '+str(math.ceil(tr_1))+'. '+str(math.ceil(tr_2))+'. '+str(math.ceil(tr_3))+'. '+str(math.ceil(tr_4))+'. '+str(math.ceil(tr_5))+'.  '+str(math.ceil(tr_6))+'. '+str(math.ceil(tr_7))+'.]')
print('-' * 25)
#input('确认结果，回车结束程序')



