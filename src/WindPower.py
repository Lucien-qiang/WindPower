import scipy.io as sio
import numpy as np
import pandas as pd
import copy
import os
import pandas as pd
from scipy import  stats
import time
import datetime
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
class WindPower:
      def __init__(self):
            self.path = '../data/windpowerdata/windpowerdata/'
            self.hisraw_columns = ['speed']
      def transferTime(self,date,second):
            d = datetime.datetime.strptime(date,"%Y%m%d")
            t = time.mktime(d.timetuple())+second
            return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(t)))
      def readData(self,subpath,filename):
            table = pd.DataFrame()
            data = sio.loadmat(subpath+filename)
            date = filename.split('.')[0]
            for col in self.hisraw_columns:
                  df = pd.DataFrame(data[col].tolist(),columns=['time_second',col])
                  df = df[df.col>0]
                  list =[]
                  x1 = 0
                  x2 = 0
                  for a in range(900,87300,900):
                        if len(df[df.time_second==a])>0:
                              left = df[df.time_second==a][0:1][col].values[0]
                              list.append([self.transferTime(date,a),left])
                        else:
                              if len(df[df.time_second<a])==0:
                                    left = df[0:1][col].values[0]

                                    x1 = df[0:1]['time_second']
                              else:
                                    left = df[df.time_second <a][-1:][col].values[0]
                                    x1 = df[0:1]['time_second']
                              if len(df[df.time_second>a])==0:
                                    right = df[-1:][col].values[0]
                                    x2 = df[-1:]['time_second']
                              else:
                                    right =  df[df.time_second>a][0:1][col].values[0]
                                    x2 = df[-1:]['time_second']
                              list.append([self.transferTime(date,a),self.linearInsertValue(x1,x2,left,right,a)])
                  df1 = pd.DataFrame(list,columns=['time_second', col])
                  table['time_second'] = df1['time_second']
                  table[col] = df1[col]
            table['date'] = date
            return table
      def readRaw(self,subpath,filename):
            table = pd.DataFrame()
            data = sio.loadmat(subpath+filename)
            df1 = pd.DataFrame(data['speed'].tolist(),columns=['time_second','speed'])
            df2 = pd.DataFrame(data['power'].tolist(),columns=['time_second','power'])
            df = pd.DataFrame(data['status'].tolist(), columns=['time_second', 'status'])
            table = pd.merge(df1,df2)
            list = []
            date = filename.split('.')[0]
            col = 'status'
            for a in table.time_second.values.tolist():
                  a = int(a)
                  timestamp = self.transferTime(date,a)
                  if len(df[df.time_second == a]) > 0:
                        left = df[df.time_second == a][0:1][col].values[0]
                        list.append([a, left,timestamp])
                  else:
                        if len(df[df.time_second < a]) == 0:
                              left = df[0:1][col].values[0]
                        else:
                              left = df[df.time_second < a][-1:][col].values[0]
                        list.append([a, left,timestamp])
            df3 = pd.DataFrame(list, columns=['time_second', col,'timestamp'])
            table['status'] = df3['status']
            table['timestamp'] = df3['timestamp']
            table = table[table.status==11]
            return table
      def readHis(self,subpath,filename):
            table = pd.DataFrame()
            data = sio.loadmat(subpath+filename)
            date = filename.split('.')[0]
            for col in self.hisraw_columns:
                  df = pd.DataFrame(data[col].tolist(),columns=['time_second',col])
                  df  =df[df.speed>0]
                  list =[]
                  for a in range(900,87300,900):
                        inter_df = df[(df.time_second>a-900)&(df.time_second<a)]
                        if len(inter_df)>0:
                              left = inter_df[col].mean()
                        else:
                              if a>900:
                                    left = list[-1][1]
                              else :
                                    left  = 0
                        list.append([self.transferTime(date,a),left])
                  df1 = pd.DataFrame(list,columns=['time_second', col])
                  table['time_second'] = df1['time_second']
                  table[col] = df1[col]
            #print table
            return table
      def datapropress(self):
            currentPath = self.path+'PredictData/'
            months = os.listdir(currentPath)
            monthdata = pd.DataFrame()
            for month in months:
                   days = os.listdir(currentPath+month+'/')
                   daydata = pd.DataFrame()
                   for day in days:
                         cols = os.listdir(currentPath+month+'/'+day+'/')
                         table = pd.DataFrame()
                         for col in cols:
                               path = currentPath+month+'/'+day+'/'+col
                               data = pd.read_table(path,header=None)
                               table[col.split('.')[0]]=data[0][89:185]
                         table['predict_date'] = month+day
                         if len(daydata)==0:
                            daydata = copy.deepcopy(table)
                         else:
                            daydata = pd.concat((daydata,table))
                   if len(monthdata)==0:
                         monthdata = copy.deepcopy(daydata)
                   else:
                         monthdata = pd.concat((monthdata, daydata))
            monthdata['Id'] = range(len(monthdata))
            monthdata.to_csv('PredictData_20161226_1.csv',index_label='subId')
      def HisRawData(self):
            currentPath = self.path+'HisRawData/'
            months = os.listdir(currentPath)
            monthdata = pd.DataFrame()
            for month in months:
                  days = os.listdir(currentPath + month + '/')
                  daydata = pd.DataFrame()
                  for day in days:
                        table = self.readHis(currentPath + month + '/',day)
                        #table = self.readRaw(currentPath + month + '/', day)
                        if len(daydata) == 0:
                              daydata = copy.deepcopy(table)
                        else:
                              daydata = pd.concat((daydata, table))
                  if len(monthdata) == 0:
                        monthdata = copy.deepcopy(daydata)
                  else:
                        monthdata = pd.concat((monthdata, daydata))
            monthdata['Id'] = range(len(monthdata))
            monthdata.to_csv('HisRawData_20161226.csv', index_label='subId')
      def joinPredict_HisRaw(self):
            #predictdata = pd.read_csv('PredictData_20161226.csv')
            hisrawdata = pd.read_csv('HisRawData_20161226.csv')
            hisrawdata = hisrawdata[96:]
            hisrawdata.to_csv('HisRawData_20160402from.csv')
            #predictdata['speed'] = hisrawdata['speed']
            #predictdata.to_csv('PredictData_v0.0.1.csv')
      def propessPredict(self):
            df = pd.read_csv('HisRawData_final.csv')
            temp = df.nspeed.sub(df.speed)
            for i in range(len(temp)):
                  if temp.ix[i]>3:
                        temp.ix[i]=3
                  elif temp.ix[i]<-3:
                        temp.ix[i]=-3
            df['speed_pre'] = df.speed.add(temp)
            df.to_csv('HisRawData_final.csv')


wp = WindPower()
#wp.propessPredict()
#wp.datapropress()
#wp.joinPredict_HisRaw()
wp.HisRawData()