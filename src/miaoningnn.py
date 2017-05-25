# -*- coding:utf-8 -*- 
#package:miaoningnn
#auther:Miaoning
#version 0.0
#Sigmoid模块
import numpy as np
class Sigmoid:
    flag=0
    def __init__(self,i):
        pass
    def forward(self,x):
        self.inp=x
        self.out=np.exp(x)/(np.exp(x)+np.exp(-x))-0.5
    def backward(self,g):
        self.gd=(2/(np.exp(self.inp)+np.exp(-self.inp))**2)*g
    def update(self,alpha):
        pass

#ReLU模块
class ReLU:
    flag=0
    def __init__(self,i):
        pass
    def forward(self,x):
        self.inp=x
        self.out=(np.sign(x)+1)/2.0*x+0.1*x
    def backward(self,g):
        self.gd=(np.sign(self.inp)+1)/2.0*g+0.1
    def update(self,alpha):
        pass

#Id模块
class Identity:
    flag=0
    def __init__(self,i):
        pass
    def forward(self,x):
        self.inp=x
        self.out=x
    def backward(self,g):
        self.gd=g
    def update(self,alpha):
        pass

#线性模块
class Linear:
    flag=1
    def __init__(self,i,o):
        self.w=np.random.random([i,o])*0.1
        self.b=np.random.random(o)*0.1
        self.gwo=np.zeros([i,o])
        self.gbo=np.zeros(o)
        self.i=i
        self.o=o
    def forward(self,x):
        self.inp=x
        self.out=np.dot(x,self.w)+self.b
    def backward(self,g):
        self.gd=np.dot(g,self.w.T)
        self.gw=np.dot(self.inp.reshape([len(self.inp),1]),g.reshape([1,len(g)])).reshape([self.i,self.o])
        self.gwo=self.gw         #加入动量算法，可以调整动量比
        self.gb=g
        self.gbo=self.gb
    def update(self,alpha,sigma=0.95):
        self.w=self.w-self.gwo*alpha
        self.b=self.b-self.gbo*alpha
        
#目标函数MSE模块
class MSE:
    def __init___(self):
        pass
    def forward(self,x,y):
        self.inp=x
        self.lable=y
        self.out=np.sum(np.square(x-y))*0.5
    def backward(self):
        self.gd=self.inp-self.lable

#容器
class Container:
    def __init__(self):
        self.layers=[]
        self.r=[]
    def addlayer(self,layer,rate=1):
        self.layers.append(layer)
        self.r.append(rate)
    def forward(self,x):
        self.inp=x
        self.layers[0].forward(x)
        for i in range(1,len(self.layers)):
            self.layers[i].forward(self.layers[i-1].out)
        self.out=self.layers[-1].out
    def backward(self,g):
        self.layers[-1].backward(g)
        for i in range(len(self.layers)-2,-1,-1):
            self.layers[i].backward(self.layers[i+1].gd)
        self.gd=self.layers[0].gd
    def update(self,alpha):
        for i in range(0,len(self.layers)):
            self.layers[i].update(alpha*self.r[i])

#平行容器
class PContainer:
    def __init__(self):
        self.par=[]
        self.dim=[]
        self.r=[]
    def addpar(self,layer,io,rate=1):
        self.par.append(layer)
        self.dim.append(io)
        self.idim=[0]
        self.odim=[0]
        self.r.append(rate)
        for item in self.dim:
            self.idim.append(self.idim[-1]+item[0])
            self.odim.append(self.odim[-1]+item[-1])
    def forward(self,x):
        self.inp=x
        self.result=[]
        for i in range(0,len(self.par)):
            self.par[i].forward(x[self.idim[i]:self.idim[i+1]])
            self.result.append(self.par[i].out)
        self.out=np.hstack(self.result)
    def backward(self,g):
        self.gdlist=[]
        for i in range(0,len(self.par)):
            self.par[i].backward(g[self.odim[i]:self.odim[i+1]])
            self.gdlist.append(self.par[i].gd)
        self.gd=np.hstack(self.gdlist)
    def update(self,alpha):
        for i in range(0,len(self.par)):
            self.par[i].update(alpha*self.r[i])

class trainer:
    def __init__(self,net1,obj1):
        self.net=net1
        self.obj=obj1
    def predict(self,data):
        if data.ndim==1:
            self.net.forward(data)
            return self.net.out
        elif data.ndim==2:
            self.result=[]
            for i in range(0,len(data)):
                self.net.forward(data[i])
                self.result.append(self.net.out)
            return np.vstack(self.result)
    def score(self,data,lable):
        if data.ndim==1:
            self.net.forward(data)
            self.obj.forward(self.net.out,lable)
            return self.obj.out
        elif data.ndim==2:
            self.err=[]
            for i in range(0,len(data)):
                self.net.forward(data[i])
                self.obj.forward(self.net.out,lable[i])
                self.err.append(self.obj.out)
            return sum(self.err)/len(self.err)
    def train(self,data,lable,alpha):
        if data.ndim==1:
            self.net.forward(data)
            self.obj.forward(self.net.out,lable)
            self.obj.backward()
            self.net.backward(self.obj.gd)
            self.net.update(alpha)
        elif data.ndim==2:
            self.err=[]
            for i in range(0,len(data)):
                self.net.forward(data[i])
                self.obj.forward(self.net.out,lable[i])
                self.obj.backward()
                self.net.backward(self.obj.gd)
                self.net.update(alpha)