
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import seaborn as sns
import scipy as sp
import math
import scipy.io as scio
import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
get_ipython().magic("config InlineBackend.figure_format = 'png' #set 'png' here when working on notebook")
get_ipython().magic('matplotlib inline')


# In[134]:

def rmse(res_y, real_y):
    n = len(res_y)
    cnt = 0.0
    for i in range(n):
        cnt += (res_y[i] - real_y[i])**2
    cnt /= n
    return math.sqrt(cnt)


# In[3]:

df = pd.read_csv('HisRawData_ForPower_Final.csv')


# In[6]:

sns.jointplot(x = 'speed', y = 'power', data = df, linewidth=0.1)


# In[7]:

df_valid = df[df['status']==11]
sum(df['status']==11)/len(df.index)


# In[8]:

sns.jointplot(x = 'speed', y = 'power', data = df_valid, linewidth=0.1)


# In[9]:

#删除power>1500的数据
print(sum(df_valid['power']>1500)/len(df_valid.index))
df_valid = df_valid[df_valid['power']<=1500]
#删除power<0的数据
print(sum(df_valid['power']<0)/len(df_valid.index))
df_valid = df_valid[df_valid['power']>=0]


# In[10]:

#df_valid['power'] = df_valid['power']**(1/3.0)


# In[11]:

#plt.scatter(list(df_valid.speed), df_valid.power)
sns.jointplot(x = 'speed', y = 'power', data = df_valid,linewidth=0.1)


# In[12]:

import re
not_july = []
t = list(df_valid['timestamp'])
for i in range(len(df_valid)):
    not_july.append(re.search(r'2016-07-',t[i]) is None)
#print(not_july)
X_train = pd.DataFrame(df_valid[not_july]['speed'])
y_train = df_valid[not_july]['power']
y_train3 = df_valid[not_july]['power']**(1/3.0)
X_july = pd.DataFrame(df_valid[np.logical_not(not_july)]['speed'])
y_test = df_valid[np.logical_not(not_july)]['power']


# In[13]:

from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, Lasso, LassoLars, LassoCV, LassoLarsCV, ElasticNet, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)

##LinearRegression
LR = LinearRegression()
LR.fit(X_train,y_train)
print('LinearRegression:', rmse(LR.predict(X_july), list(y_test))/1500)
LR.fit(X_train,y_train3)
print('LinearRegression:', rmse3(LR.predict(X_july), list(y_test))/1500)


# In[123]:

#Ridge
RG = Ridge(alpha=2.0)
RG.fit(X_train,y_train)
print('Ridge:', rmse(RG.predict(X_july), list(y_test))/1500)
RG.fit(X_train,y_train3)
print('Ridge:', rmse3(RG.predict(X_july), list(y_test))/1500)


# In[124]:

#Lasso
LS = Lasso(alpha=10)
LS.fit(X_train,y_train)
print('Lasso:', rmse(LS.predict(X_july), list(y_test))/1500)
LS.fit(X_train,y_train3)
print('Lasso:', rmse3(LS.predict(X_july), list(y_test))/1500)


# In[125]:

#RandomForest
RF = RandomForestRegressor()
RF.fit(X_train, y_train)
print('RandomForest:', rmse(RF.predict(X_july), list(y_test))/1500)
RF.fit(X_train, y_train3)
print('RandomForest:', rmse3(RF.predict(X_july), list(y_test))/1500)


# In[147]:

res = RF.predict(X_july)
xgbplt = pd.Series(res, index = list(X_july.speed))
#print(xgbplt)
#xgbplt.plot(title = "Validation")
plt.scatter(list(X_july.speed), y_test)


# In[126]:

#Xgboost
import xgboost as xgb
from sklearn import cross_validation
X_dtrain, X_deval, y_dtrain, y_deval = cross_validation.train_test_split(X_train, y_train, random_state=1026, test_size=0.2)
dtrain = xgb.DMatrix(X_dtrain, y_dtrain)
deval = xgb.DMatrix(X_deval, y_deval)
watchlist = [(deval, 'eval')]
#param_grid = {'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1], 'min_child_weight': [0.3, 0.4, 0.5, 0.6, 1], 'max_depth':[2,3,4,5,6], 
#             'max_depth':[2,3,4,5,6], 'lambda':[0,0.1,1]}
params = {
    'booster': 'gbtree',
    'objective': 'reg:linear',
    'subsample': 0.6,##
    'min_child_weight': 0.3,##
    'colsample_bytree': 1,##
    'eta': 0.02,
    'max_depth': 3,
    'seed': 2016,
    'silent': 1,
    'alpha': 0,
    'eval_metric': 'rmse'
}
clf = xgb.train(params, dtrain, 500, watchlist, early_stopping_rounds=50)
pred = clf.predict(xgb.DMatrix(X_july))
print('Xgboost:', rmse(pred, list(y_test))/1500)

X_dtrain, X_deval, y_dtrain, y_deval = cross_validation.train_test_split(X_train, y_train3, random_state=1026, test_size=0.2)
dtrain = xgb.DMatrix(X_dtrain, y_dtrain)
deval = xgb.DMatrix(X_deval, y_deval)
watchlist = [(deval, 'eval')]
clf = xgb.train(params, dtrain, 500, watchlist, early_stopping_rounds=50)
pred = clf.predict(xgb.DMatrix(X_july))
print('Xgboost:', rmse3(pred, list(y_test))/1500)


# In[133]:

preddf = pd.read_csv('predictSpeed_ForPower.csv')


# In[135]:

def rmse3(res_y, real_y):
    n = len(res_y)
    cnt = 0.0
    for i in range(n):
        cnt += (res_y[i]**3 - real_y[i])**2
    cnt /= n
    return math.sqrt(cnt)


# In[136]:

preddf = preddf[preddf['status']==11]
#preddf = preddf[preddf['power'] >=0]
preddf = preddf[preddf['power'] <=1500]


# In[137]:

X_pred = pd.DataFrame(preddf['speed'])
y_real = preddf['power']


# In[138]:

from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, Lasso, LassoLars, LassoCV, LassoLarsCV, ElasticNet, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)

##LinearRegression
LR = LinearRegression()
LR.fit(X_train,y_train)
print('LinearRegression:', rmse(LR.predict(X_pred), list(y_real))/1500)
LR.fit(X_train,y_train3)
print('LinearRegression:', rmse3(LR.predict(X_pred), list(y_real))/1500)


# In[139]:

#Ridge
RG = Ridge(alpha=2.0)
RG.fit(X_train,y_train)
print('Ridge:', rmse(RG.predict(X_pred), list(y_real))/1500)
RG.fit(X_train,y_train3)
print('Ridge:', rmse3(RG.predict(X_pred), list(y_real))/1500)


# In[140]:

#Lasso
LS = Lasso(alpha=10)
LS.fit(X_train,y_train)
print('Lasso:', rmse(LS.predict(X_pred), list(y_real))/1500)
LS.fit(X_train,y_train3)
print('Lasso:', rmse3(LS.predict(X_pred), list(y_real))/1500)


# In[141]:

#RandomForest
RF = RandomForestRegressor()
RF.fit(X_train, y_train)
print('RandomForest:', rmse(RF.predict(X_pred), list(y_real))/1500)
RF.fit(X_train, y_train3)
print('RandomForest:', rmse3(RF.predict(X_pred), list(y_real))/1500)


# In[142]:

#Xgboost
import xgboost as xgb
from sklearn import cross_validation
X_dtrain, X_deval, y_dtrain, y_deval = cross_validation.train_test_split(X_train, y_train, random_state=1026, test_size=0.2)
dtrain = xgb.DMatrix(X_dtrain, y_dtrain)
deval = xgb.DMatrix(X_deval, y_deval)
watchlist = [(deval, 'eval')]
#param_grid = {'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1], 'min_child_weight': [0.3, 0.4, 0.5, 0.6, 1], 'max_depth':[2,3,4,5,6], 
#             'max_depth':[2,3,4,5,6], 'lambda':[0,0.1,1]}
params = {
    'booster': 'gbtree',
    'objective': 'reg:linear',
    'subsample': 0.6,##
    'min_child_weight': 0.3,##
    'colsample_bytree': 1,##
    'eta': 0.02,
    'max_depth': 3,
    'seed': 2016,
    'silent': 1,
    'alpha': 0,
    'eval_metric': 'rmse'
}
clf = xgb.train(params, dtrain, 500, watchlist, early_stopping_rounds=50)
pred = clf.predict(xgb.DMatrix(X_pred))
print('Xgboost:', rmse(pred, list(y_real))/1500)

X_dtrain, X_deval, y_dtrain, y_deval = cross_validation.train_test_split(X_train, y_train3, random_state=1026, test_size=0.2)
dtrain = xgb.DMatrix(X_dtrain, y_dtrain)
deval = xgb.DMatrix(X_deval, y_deval)
watchlist = [(deval, 'eval')]
clf = xgb.train(params, dtrain, 500, watchlist, early_stopping_rounds=50)
pred = clf.predict(xgb.DMatrix(X_pred))
print('Xgboost:', rmse3(pred, list(y_real))/1500)


# In[20]:

alphas = [0.01+i*0.01 for i in range(10)]
cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() for alpha in alphas]
cv_ridge = pd.Series(cv_ridge, index = alphas)
print(cv_ridge)
cv_ridge.plot(title = "Validation")
plt.xlabel("alpha")
plt.ylabel("rmse")
plt.show()
print('Ridge:',cv_ridge.min())


# In[23]:

import xgboost as xgb

regr = xgb.XGBRegressor(colsample_bytree=1, gamma=0.0, learning_rate=0.006, max_depth=2,min_child_weight=1,
                        n_estimators=7200, reg_alpha=0.9,reg_lambda=0.6,subsample=0.5,seed=42,silent=1)
regr.fit(X_train, y_train)
print(rmse(list(regr.predict(X_test)), list(y_test))/1500)


# In[95]:

import xgboost as xgb
from sklearn import cross_validation
X_dtrain, X_deval, y_dtrain, y_deval = cross_validation.train_test_split(X_train, y_train3, random_state=1026, test_size=0.2)
dtrain = xgb.DMatrix(X_dtrain, y_dtrain)
deval = xgb.DMatrix(X_deval, y_deval)
watchlist = [(deval, 'eval')]
param_grid = {'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1], 'min_child_weight': [0.3, 0.4, 0.5, 0.6, 1], 'max_depth':[2,3,4,5,6], 
             'max_depth':[2,3,4,5,6], 'lambda':[0,0.1,1]}


params = {
    'booster': 'gbtree',
    'objective': 'reg:linear',
    'subsample': 0.8,##
    'min_child_weight': 0.8,##
    'colsample_bytree': 1,##
    'eta': 0.02,
    'max_depth': 3,
    'seed': 2016,
    'silent': 1,
    'alpha': 0,
    'eval_metric': 'rmse'
}
clf = xgb.train(params, dtrain, 500, watchlist, early_stopping_rounds=50)
pred = clf.predict(xgb.DMatrix(X_pred))


# In[96]:

print('Xgboost:', rmse3(list(pred), list(y_real))/1500)


# In[516]:

import xgboost as xgb
from sklearn import cross_validation

from sklearn.model_selection import GridSearchCV

param_grid = {'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1], 'min_child_weight': [0.3, 0.4, 0.5, 0.6, 1], 'max_depth':[2,3,4,5,6]}
svr = xgb.XGBRegressor(objective='reg:linear', learning_rate=0.1,
    reg_alpha=0,reg_lambda=0.6, seed=42,silent=1)
clf = GridSearchCV(svr, param_grid)
clf.fit(pd.DataFrame(df_valid['speed']), df_valid['power'])
xgb.XGBRegressor()


# In[517]:

print(clf.best_params_)


# In[610]:

pred = clf.predict(xgb.DMatrix(speed_pree))


# In[ ]:

import xgboost as xgb
from sklearn import cross_validation
X_dtrain, X_deval, y_dtrain, y_deval = cross_validation.train_test_split(X_train, y_train, random_state=1026, test_size=0.2)
dtrain = xgb.DMatrix(X_dtrain, y_dtrain)
deval = xgb.DMatrix(X_deval, y_deval)
watchlist = [(deval, 'eval')]
#param_grid = {'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1], 'min_child_weight': [0.3, 0.4, 0.5, 0.6, 1],
#              'max_depth':[2,3,4,5,6], 'lambda':[0,0.1,1]}
param_grid = {'subsample': [0.6,0.8], 'min_child_weight': [0.4,0.5,0.6,0.7,0.8,0.9,1],
              'max_depth':[2], 'lambda':[0,0.1]}
resultstr=[]
for sub in param_grid['subsample']:
    for min_child in param_grid['min_child_weight']:
        for max_d in param_grid['max_depth']:
            for lam in param_grid['lambda']:  
                params = {
                    'booster': 'gbtree',
                    'objective': 'reg:linear',
                    'subsample': sub,
                    'min_child_weight': min_child,
                    'colsample_bytree': 1,
                    'eta': 0.1,
                    'max_depth': max_d,
                    'seed': 2016,
                    'silent': 1,
                    'alpha': 0,
                    'lambda': lam,
                    'eval_metric': 'rmse'
                }
                clf = xgb.train(params, dtrain, 500, watchlist, early_stopping_rounds=50)
                pred = clf.predict(xgb.DMatrix(speed_pree))
                resultstr.append('sub:'+str(sub)+',min_child:'+str(min_child)+',max_d:'+str(max_d)+',lam:'+str(lam))
                resultstr.append(rmse(list(pred), list(powerr))/1500)


# In[ ]:



