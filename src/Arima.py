# -*-coding:utf-8-*-
import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARMA
import sys
from dateutil.relativedelta import relativedelta
from copy import deepcopy
import matplotlib.pyplot as plt

class arima_model:
    def __init__(self, ts, maxLag=3):
        self.data_ts = ts
        self.resid_ts = None
        self.predict_ts = None
        self.maxLag = maxLag
        self.p = maxLag
        self.q = maxLag
        self.properModel = None
        self.bic = sys.maxint

    # 计算最优ARIMA模型，将相关结果赋给相应属性
    def get_proper_model(self):
        self._proper_model()
        self.predict_ts = deepcopy(self.properModel.predict())
        self.resid_ts = deepcopy(self.properModel.resid)

    # 对于给定范围内的p,q计算拟合得最好的arima模型，这里是对差分好的数据进行拟合，故差分恒为0
    def _proper_model(self):
        for p in np.arange(self.maxLag):
            for q in np.arange(self.maxLag):
                # print p,q,self.bic
                model = ARMA(self.data_ts, order=(p, q))
                try:
                    results_ARMA = model.fit(disp=-1, method='css')
                except:
                    continue
                bic = results_ARMA.bic
                # print 'bic:',bic,'self.bic:',self.bic
                if bic < self.bic:
                    self.p = p
                    self.q = q
                    self.properModel = results_ARMA
                    self.bic = bic
                    self.resid_ts = deepcopy(self.properModel.resid)
                    self.predict_ts = self.properModel.predict()

    # 参数确定模型
    def certain_model(self, p, q):
            model = ARMA(self.data_ts, order=(p, q))
            try:
                self.properModel = model.fit( disp=-1, method='css')
                self.p = p
                self.q = q
                self.bic = self.properModel.bic
                self.predict_ts = self.properModel.predict()
                self.resid_ts = deepcopy(self.properModel.resid)
            except:
                print 'You can not fit the model with this parameter p,q, ' \
                      'please use the get_proper_model method to get the best model'

    # 预测第二日的值
    def forecast_next_day_value(self, type='quarter'):
        # 我修改了statsmodels包中arima_model的源代码，添加了constant属性，需要先运行forecast方法，为constant赋值
        self.properModel.forecast()
        if self.data_ts.index[-1] != self.resid_ts.index[-1]:
            raise ValueError('''The index is different in data_ts and resid_ts, please add new data to data_ts.
            If you just want to forecast the next day data without add the real next day data to data_ts,
            please run the predict method which arima_model included itself''')
        if not self.properModel:
            raise ValueError('The arima model have not computed, please run the proper_model method before')
        para = self.properModel.params

        # print self.properModel.params
        if self.p == 0:   # It will get all the value series with setting self.data_ts[-self.p:] when p is zero
            ma_value = self.resid_ts[-self.q:]
            values = ma_value.reindex(index=ma_value.index[::-1])
        elif self.q == 0:
            ar_value = self.data_ts[-self.p:]
            values = ar_value.reindex(index=ar_value.index[::-1])
        else:
            ar_value = self.data_ts[-self.p:]
            ar_value = ar_value.reindex(index=ar_value.index[::-1])
            ma_value = self.resid_ts[-self.q:]
            ma_value = ma_value.reindex(index=ma_value.index[::-1])
            values = ar_value.append(ma_value)

        #predict_value = np.dot(para[1:], values) + self.properModel.constant[0]
        predict_value = np.dot(para[1:], values)
        self._add_new_data(self.predict_ts, predict_value, type)
        return predict_value

    # 动态添加数据函数，针对索引是月份和日分别进行处理
    def _add_new_data(self, ts, dat, type='quarter'):
        if type == 'day':
            new_index = ts.index[-1] + relativedelta(days=1)
        elif type == 'quarter':
            new_index = ts.index[-1] + relativedelta(minutes=15)
        ts[new_index] = dat
    def add_today_data(self, dat, type='quarter'):
        self._add_new_data(self.data_ts, dat, type)
        if self.data_ts.index[-1] != self.predict_ts.index[-1]:
            raise ValueError('You must use the forecast_next_day_value method forecast the value of today before')
        self._add_new_data(self.resid_ts, self.data_ts[-1] - self.predict_ts[-1], type)

# 差分操作
def diff_ts(ts, d):
    global shift_ts_list
    #  动态预测第二日的值时所需要的差分序列
    global last_data_shift_list
    shift_ts_list = []
    last_data_shift_list = []
    tmp_ts = ts
    for i in d:
        last_data_shift_list.append(tmp_ts[-i])
        shift_ts = tmp_ts.shift(i)
        shift_ts_list.append(shift_ts)
        tmp_ts = tmp_ts - shift_ts
    tmp_ts.dropna(inplace=True)
    return tmp_ts

# 还原操作
def predict_diff_recover(predict_value, d):
    if isinstance(predict_value, float):
        tmp_data = predict_value
        for i in range(len(d)):
            tmp_data = tmp_data + last_data_shift_list[-i-1]
    elif isinstance(predict_value, np.ndarray):
        tmp_data = predict_value[0]
        for i in range(len(d)):
            tmp_data = tmp_data + last_data_shift_list[-i-1]
    else:
        tmp_data = predict_value
        for i in range(len(d)):
            try:
                tmp_data = tmp_data.add(shift_ts_list[-i-1])
            except:
                raise ValueError('What you input is not pd.Series type!')
        tmp_data.dropna(inplace=True)
    return tmp_data

def proper_model(data_ts, maxLag):
    init_bic = sys.maxint
    init_p = 0
    init_q = 0
    init_properModel = None
    for p in np.arange(maxLag):
        for q in np.arange(maxLag):
            model = ARMA(data_ts, order=(p, q))
            try:
                results_ARMA = model.fit(disp=-1, method='css')
            except:
                continue
            bic = results_ARMA.bic
            if bic < init_bic:
                init_p = p
                init_q = q
                init_properModel = results_ARMA
                init_bic = bic
    return init_bic, init_p, init_q, init_properModel

from dateutil.relativedelta import relativedelta
def _add_new_data(ts, dat, type='quarter'):
    if type == 'day':
        new_index = ts.index[-1] + relativedelta(days=1)
    elif type == 'quarter':
        new_index = ts.index[-1] + relativedelta(minutes=15)
    ts[new_index] = dat

def add_today_data(model, ts,  data, d, type='quarter'):
    _add_new_data(ts, data, type)  # 为原始序列添加数据
    # 为滞后序列添加新值
    d_ts = diff_ts(ts, d)
    model.add_today_data(d_ts[-1], type)

def forecast_next_day_data(model, type='quarter'):
    if model == None:
        raise ValueError('No model fit before')
    fc = model.forecast_next_day_value(type)
    return predict_diff_recover(fc, [288, 1])
import time
import datetime
def getC(df,day):
      ts = df['speed']
      tx = df['NTspeed_Corrected']
      ts_log = ts
      ts_log.index = pd.to_datetime(ts_log.index)
      ts_train = ts_log[:day]
      ts_test = ts_log[day:(day+relativedelta(days=1))+relativedelta(hours=12)]
      tx_test = tx[day:(day+relativedelta(days=1))+relativedelta(hours=12)]
      diffed_ts = diff_ts(ts_train, [288, 1])
      forecast_list = []
      for i, dta in enumerate(ts_test):
            if i % 96 == 0:
                  model = arima_model(diffed_ts)
                  model.certain_model(1, 1)
            forecast_data = forecast_next_day_data(model, type='quarter')
            forecast_list.append(forecast_data)
            add_today_data(model, ts_train, tx_test[i], [288, 1], type='quarter')
      print (day+relativedelta(days=1))+relativedelta(hours=12)
      predict_ts = pd.Series(data=forecast_list, index=ts[day:day+relativedelta(days=1)+relativedelta(hours=12)].index)
      log_recover = predict_ts
      ts = ts[day:day+relativedelta(days=1)+relativedelta(hours=12)]
      final = log_recover.add(tx_test)
      #final = log_recover
      final = final.mul(0.5)
      final = final.add(0)
      plt.figure(facecolor='white')
      #log_recover.plot(color='blue', label='Predict')
      tx_test.plot(color='green', label='Nspeed')
      ts.plot(color='blue', label='Original')
      final.plot(color='red', label='Predict')
      plt.legend(loc='best')
      rmse = np.sqrt(sum((log_recover[day+relativedelta(hours=12):day+relativedelta(days=1)+relativedelta(hours=12)] -
                          ts[day+relativedelta(hours=12):day+relativedelta(days=1)+relativedelta(hours=12)]) ** 2) / 97) / 14
      print ('RMSE0: %.5f' % rmse)
      rmse1 = np.sqrt(sum((tx_test[day+relativedelta(hours=12):day+relativedelta(days=1)+relativedelta(hours=12)] -
                           ts[day+relativedelta(hours=12):day+relativedelta(days=1)+relativedelta(hours=12)]) ** 2) / 97) / 14
      print ('RMSE1: %.5f' % rmse1)
      rmse2 = np.sqrt(sum((final[day+relativedelta(hours=12):day+relativedelta(days=1)+relativedelta(hours=12)] -
                           ts[day+relativedelta(hours=12):day+relativedelta(days=1)+relativedelta(hours=12)]) ** 2) / 97) / 14
      plt.title('RMSE: %.5f' % rmse2)
      plt.show()

      return [day,rmse1,rmse,rmse2],final[day+relativedelta(hours=12)+relativedelta(minutes=15):day+relativedelta(days=1)+relativedelta(hours=12)]

if __name__ == '__main__':
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d %H:%M:%S')
    df = pd.read_csv('D:/pyplace/WindPower/PredictData_20161226.csv', parse_dates='time_second', index_col='time_second',
                        date_parser=dateparse)
    rmselist = []
    data = pd.DataFrame()
    day = datetime.datetime.strptime('2016-06-29 12:00:00', "%Y-%m-%d %H:%M:%S")

    for i in range(31):
          day = day + relativedelta(days=1)
          sublist,sybdata = getC(df,day)
          rmselist.append(sublist)
          if i==0:
                data = sybdata
          else:
                data = pd.concat((data,sybdata))
    (pd.DataFrame(rmselist,columns=['day','nt_rmse','arima_rmse','final_rmse'])).to_csv('final_rmse1.csv')
    data.to_csv('final_speed1.csv')



