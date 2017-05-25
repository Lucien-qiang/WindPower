
# coding: utf-8

# In[ ]:

import math
import numpy as np


# In[ ]:

f1=open('predictspeed1')
f2=open('predictspeed2')
f3=open('predictt')
f4=open('predictp')
f5=open('predictu')
f6=open('predictv')
f7=open('predictw')
f=[f1,f2,f3,f4,f5,f6,f7]
g=open('realspeed')


# In[ ]:

import numpy as np
import re


# In[ ]:

data1=[]
for line in f[0]:
    L=[]
    L.append(line)
    for i in range(1,len(f)):
        L.append(f[i].readline())
    tem=[]
    item2=L[0].split('\t')
    tem.append(int(item2[0]))
    tem.append(int(item2[1]))
    tem.append(float(item2[2]))
    for i in range(1,len(L)):
        item2=L[i].split('\t')
        tem.append(float(item2[2]))
    if tem[1]<96:
        data1.append(tem)


# In[ ]:

data2=[]
for line in g:
    item2=line.split('\t')
    tem=[int(item2[0]),int(float(item2[1])/900+0.5), float(item2[2])]
    if tem[2]!=0:
        data2.append(tem)


# In[ ]:

def compare(A,B):
    if A[0]>B[0]:
        return 1
    elif A[0]<B[0]:
        return -1
    else:
        if A[1]>B[1]:
            return 1
        elif A[1]<B[1]:
            return -1
        else:
            return 0


# In[ ]:

j=0
data3=[]
for i in range(0,len(data1)):
    tem1=[]
    for item in data1[i]:
        tem1.append(item)
    tem=[]
    while True:
        c=compare(data2[j][0:2],data1[i][0:2])
        if c==-1:
            j=j+1
            continue
        if c==0:
            tem.append(data2[j][2])
            j=j+1
            continue
        if c==1:
            if tem==[]:
                tem1.append(0)
                break
            else:
                tem1.append(sum(tem)/len(tem))
                break
    data3.append(tem1)

flag=0
i=0
for item in data3:
    if item[-1]==0:
        i=i+1
        if item[0]!=flag:
            flag=item[0]
            print flag,i
# In[ ]:

def date2num(s):
    s=s-(s/10000)*10000
    s1=s/100
    s2=s-s1*100
    if s1==4:
        return s2-1
    if s1==5:
        return s2+29
    if s1==6:
        return s2+29+31
    if s1==7:
        return s2+29+31+30


# In[ ]:

data4=[]
for item in data3:
    tem=[]
    for i in item:
        tem.append(i)
    tem[0]=date2num(tem[0])
    data4.append(tem)


# In[ ]:

data5=np.array(data4)


# In[ ]:

N0=[]
for i in range(0,len(data5)):
    if data5[i][-1]>0:
        N0.append(i)


# In[ ]:

data6=data5[N0]


# In[ ]:

data7=[]
p=-1
a=[[],[],[]]
for line in data6:
    p=p+1
    if p%100==0:
        print p
    tem=[]
    flag=0
    tem.append(line[0])
    tem.append(math.sin(2*math.pi*line[1]/96.0))
    tem.append(math.cos(2*math.pi*line[1]/96.0))
    #前一天的各小时平均风速，共15项
    tem2=[]
    for i in range(0,15):
        tem2.append([])
    for line2 in data6:
        if line2[0]==line[0]-1:
            h=int(line2[1]/4)
            if h<15:
                tem2[h].append(line2[-1])
    for item in tem2:
        if item==[]:
            flag=1
            a[0].append(p)
            break
        else:
            tem.append(sum(item)/(len(item)+0.0))
    #前七天平均风速，共7项
    dd=3
    tem3=[]
    for i in range(0,dd):
        tem3.append([])
    for line3 in data6:
        c=line[0]-line3[0]
        if c<=dd and c>0:
            c=int(c)-1
            tem3[c].append(line3[-1])
    for item in tem3:
        if len(item)<50:
            flag=1
            a[1].append(p)
            break
        else:
            tem.append(sum(item)/(len(item)+0.0))
    #前七天瞬时风速，共7项
    tem4=[]
    for i in range(0,dd):
        tem4.append([])
    for line4 in data6:
        c=line[0]-line4[0]
        if c<=dd and c>0:
            if line4[1]==line[1]:
                c=int(c)-1
                tem4[c].append(line4[-1])
    for item in tem4:
        if item==[]:
            flag=1
            a[2].append(p)
            break
        else:
            tem.append(item[0])
    for i in range(2,10):
        tem.append(line[i])
    if flag==0:
        data7.append(tem)

tem=[0,0]
for i in range(0,len(data6)):
    if tem[1]==95:
        if data6[i][0]!=tem[0]+1 or data6[i][1]!=0:
            print i
    else:
        if data6[i][0]!=tem[0] or data6[i][1]!=tem[1]+1:
            print i
    tem=[data6[i][0],data6[i][1]]
# In[ ]:

data75=np.array(data7)


# In[ ]:

data8=data75.copy()
for i in range(0,len(data8)):
    tem=[]
    tem.append(data8[i][24])
    tem.append(data8[i][25])
    data8[i][24]=data8[i][29]
    data8[i][25]=data8[i][30]
    data8[i][29]=tem[0]
    data8[i][30]=tem[1]


# In[ ]:

data85=[]
for item in data8:
    tem=[]
    for i in range(0,len(item)-1):
        tem.append(item[i])
    tem.append(item[-3])
    tem.append(item[-2])
    tem.append(item[-1])
    data85.append(tem)
data9=np.array(data85)


# In[ ]:

def convert(data):
    sta=[]
    datan=data.copy()
    l1=len(data)
    l2=len(data[0])
    for i in range(0,l2):
        tem=[]
        tem.append(datan[:,i].mean())
        tem.append(datan[:,i].std())
        datan[:,i]=datan[:,i]-tem[0]
        datan[:,i]=datan[:,i]/tem[1]
        sta.append(tem)
    return datan,sta
def invconvert(data,sta):
    datan=data.copy()
    l1=len(data)
    l2=len(data[0])
    for i in range(0,l2):
        datan[:,i]=datan[:,i]*sta[i,1]
        datan[:,i]=datan[:,i]+sta[i,0]
    return datan


# In[ ]:

data10,sta=convert(data9)


# In[ ]:

data=data10
#r=range(0,len(data))
#np.random.shuffle(r)
#traindata=data[r[:int(len(data)*0.8)],:]
#testdata=data[r[int(len(data)*0.8):],:]
trainset=[]
testset=[]
for i in range(0,len(data)):
    if data[i][0]*sta[0][1]+sta[0][0]<92:
        trainset.append(i)
    else:
        testset.append(i)
traindata=data[trainset,:]
testdata=data[testset,:]
trainx=traindata[:,0:-1]
trainy=traindata[:,[-1]]
testx=testdata[:,0:-1]
testy=testdata[:,[-1]]

trainx2=trainx.copy()
testx2=testx.copy()
for i in range(0,len(trainx)):
    for j in range(len(trainx[0])-2,len(trainx[0])):
        trainx2[i][j]=0
for i in range(0,len(testx)):
    for j in range(len(trainx[0])-2,len(trainx[0])):
        testx2[i][j]=0
# In[3]:

from miaoningnn import ReLU
from miaoningnn import Container
from miaoningnn import Sigmoid
from miaoningnn import Identity
from miaoningnn import PContainer
from miaoningnn import Linear
from miaoningnn import MSE
from miaoningnn import trainer


# In[5]:

#build network
nn=Container()

p1=PContainer()

p=PContainer()

nn.addlayer(p1)

nn2=Container()

p1.addpar(nn2,[31,8],rate=1)

nn2.addlayer(p)
nn2.addlayer(Linear(35,50))
nn2.addlayer(Sigmoid(50))
nn2.addlayer(Linear(50,8))

nn2.addlayer(Sigmoid(8))

nnp1=Container()
p.addpar(nnp1,[1,5])
nnp1.addlayer(Linear(1,10))
nnp1.addlayer(Sigmoid(10))
nnp1.addlayer(Linear(10,5))
nnp1.addlayer(Sigmoid(5))

nnp2=Container()
p.addpar(nnp2,[2,5])
nnp2.addlayer(Linear(2,10))
nnp2.addlayer(Sigmoid(10))
nnp2.addlayer(Linear(10,5))
nnp2.addlayer(Sigmoid(5))

nnp3=Container()
p.addpar(nnp3,[15,5])
nnp3.addlayer(Linear(15,10))
nnp3.addlayer(Sigmoid(10))
nnp3.addlayer(Linear(10,5))
nnp3.addlayer(Sigmoid(5))

nnp4=Container()
p.addpar(nnp4,[3,5])
nnp4.addlayer(Linear(3,10))
nnp4.addlayer(Sigmoid(10))
nnp4.addlayer(Linear(10,5))
nnp4.addlayer(Sigmoid(5))

nnp5=Container()
p.addpar(nnp5,[3,5])
nnp5.addlayer(Linear(3,10))
nnp5.addlayer(Sigmoid(10))
nnp5.addlayer(Linear(10,5))
nnp5.addlayer(Sigmoid(5))

nnp7=Container()
p.addpar(nnp7,[5,5])
nnp7.addlayer(Linear(5,10))
nnp7.addlayer(Sigmoid(10))
nnp7.addlayer(Linear(10,5))
nnp7.addlayer(Sigmoid(5))

nnp8=Container()
p.addpar(nnp8,[2,5])
nnp8.addlayer(Linear(2,5))
nnp8.addlayer(ReLU(5))

nnp6=Container()
p1.addpar(nnp6,[2,2])
nnp6.addlayer(Identity(2))

nn.addlayer(Linear(10,1))
T=trainer(nn,MSE())


# In[6]:

#train
print T.score(trainx,trainy)
print T.score(testx,testy)
for i in range(0,10):
    T.train(trainx,trainy,0.0010)
    print 'train',i,T.score(trainx,trainy)
    print 'test',i,T.score(testx,testy)


# In[ ]:



