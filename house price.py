import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.ensemble import GradientBoostingRegressor
from xgboost.sklearn import XGBRegressor
from lightgbm.sklearn import LGBMRegressor
from sklearn.model_selection import GridSearchCV
from scipy.stats import norm
from scipy import stats
#from sklearn.preprocessing import RobustScaler
#from sklearn import preprocessing
from scipy.stats import skew
from sqlalchemy import create_engine 

engine = create_engine('mysql+mysqlconnector://root:password@localhost:3306/mywork')
qry="select * from house_price"
data= pd.read_sql_query(qry,engine)

data = pd.read_csv('E:/College/Analytics/Python/house-prices/train.csv')
data.shape
data.info()
data.head(5)
data.isnull().sum().sort_values(ascending=False)

data['SalePrice'].describe()
#data['SaleCondition'].unique()
data.groupby('SaleCondition')['SalePrice'].sum().sort_values(ascending=False)
data.groupby('YrSold')['SalePrice'].sum().sort_values(ascending=False)
data.groupby('Neighborhood')['SalePrice'].sum().sort_values(ascending=False)
data.groupby('MSZoning')['SalePrice'].sum().sort_values(ascending=False)

sns.distplot(data['SalePrice'])
sns.distplot(data['SalePrice'], fit=norm)
stats.probplot(data['SalePrice'], plot=plt)
sns.distplot(np.log(data['SalePrice']), fit=norm)

data["SalePrice"] = np.log1p(data["SalePrice"])

plt.scatter(data=data,  x='GrLivArea', y='SalePrice',color='crimson', alpha=0.5)
plt.ylim(0,800000)
plt.ylabel('SalePrice', fontsize=12)
plt.xlabel('GrLivArea', fontsize=12)

data = data.drop(data[(data['GrLivArea']>4000) & (data['SalePrice']<250000)].index)

data[(data['MasVnrArea']== "NA")].count()

sns.factorplot('FireplaceQu', 'SalePrice', data = data, estimator = np.median, order = ['Ex', 'Gd', 'TA', 'Fa', 'Po'], size = 4,  aspect=1.35)
sns.boxplot(x='OverallQual',y='SalePrice', data=data)
sns.boxplot(x='MSZoning',y='SalePrice', data=data)
sns.boxplot(x='YrSold',y='SalePrice', data=data)
plt.subplots(figsize=(20, 8))
sns.boxplot(x='Neighborhood',y='SalePrice', data=data)

plt.figure(figsize = (12,8))
sns.distplot(data.kurt(),color='r',axlabel ='Kurtosis',norm_hist= False, kde = True,rug = False)

#correlation matrix
corrmat = data.corr()
corrmat.sort_values(["SalePrice"], ascending = False, inplace = True)
print(corrmat.SalePrice)
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)

#saleprice correlation matrix
k = 11 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(data[cols].values.T)
#sns.set(font_scale=1.25)
plt.subplots(figsize=(7,7))
hm = sns.heatmap(cm, annot=True, square=True,cmap='viridis',linecolor="white", annot_kws={'size': 12}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

#Data Cleaning
data["MasVnrArea"]=data["MasVnrArea"].apply(lambda x : x.replace('NA', '0')).astype(np.int64)
data["LotFrontage"]=data["LotFrontage"].apply(lambda x : x.replace('NA', '0')).astype(np.int64)

data['Electrical'] = data['Electrical'].fillna(data['Electrical'].mode()[0])
data["LotFrontage"] = data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))
data["MasVnrArea"] = data["MasVnrArea"].fillna(0)
#data["GarageYrBlt"] = data["GarageYrBlt"].fillna(0)
#Categorical missing values
NAcols=data.columns
for col in NAcols:
    if data[col].dtype == "object":
        data[col] = data[col].fillna("None")

for col in ('Alley','FireplaceQu','PoolQC','MiscFeature','Fence','MasVnrType','GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    data[col] = data[col].fillna('None')
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    data[col] = data[col].fillna('None')

data.isnull().sum().sort_values(ascending=False)

data.groupby('Utilities')['Utilities'].count()
data.groupby('PoolQC')['PoolQC'].count()
data.groupby('CentralAir')['CentralAir'].count()

data.drop (columns =['Id','1stFlrSF','OverallQual','PoolArea','PoolQC','Utilities','GarageArea','TotRmsAbvGrd','GarageYrBlt'], inplace = True)

data['MSSubClass'] = data['MSSubClass'].astype(str)
data['OverallCond'] = data['OverallCond'].astype(str)
data['YrSold'] = data['YrSold'].astype(str)
data['YearBuilt'] = data['YearBuilt'].astype(str)
data['MoSold'] = data['MoSold'].astype(str)
data['YearRemodAdd'] = data['YearRemodAdd'].astype(str)

data["MasVnrArea"]=data["MasVnrArea"].astype(np.int64)
data["LotFrontage"]=data["LotFrontage"].astype(np.int64)
data.info()
data.kurt().sort_values(ascending=False)

num_feat = data.dtypes[data.dtypes != "object"].index
skewness = data[num_feat].apply(lambda x: skew(x))
skewness = skewness[abs(skewness) > 10]
print(str(skewness.shape[0]) + " skewed numerical features to log transform")
skewed_features = skewness.index
data[skewed_features] = np.log1p(data[skewed_features])

'''
from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    #all_data[feat] += 1
    data[feat] = boxcox1p(data[feat], lam)

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
cat_feature = data.dtypes == object
cat_cols = data.columns[cat_feature].tolist()
le=LabelEncoder()
data[cat_cols] = data[cat_cols].apply(lambda col:le.fit_transform(col))

ohenc = OneHotEncoder(categorical_features=cat_feature,sparse=False)
data_ohe =ohenc.fit_transform(data)
data_ohe.shape
X=data_ohe.drop(['SalePrice'], axis=1).values
y=data_ohe['SalePrice']

X=RobustScaler(quantile_range=(25, 75)).fit_transform(X)
X=preprocessing.StandardScaler(X)
'''

df = pd.get_dummies(data,prefix_sep='_', drop_first=True)
df.shape
X=df.drop(['SalePrice'], axis=1)
y=df['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 10)

lr= LinearRegression()
lr.fit(X_train, y_train)
print(lr.intercept_)
lr_pred = lr.predict(X_test)
accuracy = lr.score(X_test,y_test)
'Accuracy: ' + str(np.round(accuracy*100, 2)) + '%'
mean_absolute_error(y_test, lr_pred)
mean_squared_error(y_test, lr_pred)
np.sqrt(mean_squared_error(y_test, lr_pred))

parameters= {'alpha':[0.0001,0.0009,0.001,0.003,0.01,0.1,1, 3, 5]}

la=Lasso(random_state=96)
la_gd=GridSearchCV(la,param_grid=parameters,scoring='neg_mean_squared_error',cv=10)
la_gd.fit(X_train, y_train)
la_gd.best_params_

la=Lasso(alpha=0.0009,random_state=96)
la.fit(X_train, y_train)
la_pred = la.predict(X_test)
accuracy = la.score(X_test,y_test)
'Accuracy: ' + str(np.round(accuracy*100, 2)) + '%'
mean_absolute_error(y_test, la_pred)
mean_squared_error(y_test, la_pred)
np.sqrt(mean_squared_error(y_test, la_pred))

coefs = pd.Series(la.coef_, index = X_train.columns)
imp_coefs = pd.concat([coefs.sort_values().head(10),
                     coefs.sort_values().tail(10)])
imp_coefs.plot(kind = "barh", color='yellowgreen')
plt.xlabel("Lasso coefficient", weight='bold')
plt.title("Feature importance in the Lasso Model", weight='bold')
plt.show()

para= {'alpha':[x for x in range(1,101)]}
rid=Ridge(random_state=96)
rid_gd=GridSearchCV(rid,param_grid=para,cv=10)
rid_gd.fit(X_train, y_train)
rid_gd.best_params_

rid=Ridge(alpha=14,random_state=96)
rid.fit(X_train, y_train)
rid_pred = rid.predict(X_test)
accuracy = rid.score(X_test,y_test)
'Accuracy: ' + str(np.round(accuracy*100, 2)) + '%'
mean_absolute_error(y_test, rid_pred)
mean_squared_error(y_test, rid_pred)
np.sqrt(mean_squared_error(y_test, rid_pred))

coefs = pd.Series(rid.coef_, index = X_train.columns)
imp_coefs = pd.concat([coefs.sort_values().head(10),coefs.sort_values().tail(10)])
imp_coefs.plot(kind = "barh")
plt.xlabel("Ridge coefficient", weight='bold')
plt.title("Feature importance in the Ridge Model", weight='bold')
plt.show()

gboost= GradientBoostingRegressor()
gboost.fit(X_train, y_train)
gboost_pred = gboost.predict(X_test)
accuracy = gboost.score(X_test,y_test)
'Accuracy: ' + str(np.round(accuracy*100, 2)) + '%'
mean_absolute_error(y_test, gboost_pred)
mean_squared_error(y_test, gboost_pred)
np.sqrt(mean_squared_error(y_test, gboost_pred))

grid= {"min_child_weight":[4,5,6],"max_depth":[3,5,7], "colsample_bytree":[0.6,0.7,0.8]}
xg=XGBRegressor(random_state=96,objective ='reg:linear')
gridsearch= GridSearchCV(xg,param_grid=grid,cv=5)
gridsearch.fit(X_train, y_train)
print(gridsearch.best_score_)
print(gridsearch.best_params_)

xgb= XGBRegressor(random_state=96,objective ='reg:linear',min_child_weight=6,n_estimators=1000,max_depth=7,colsample_bytree=0.6)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)
accuracy = xgb.score(X_test,y_test)
'Accuracy: ' + str(np.round(accuracy*100, 2)) + '%'
mean_absolute_error(y_test, xgb_pred)
mean_squared_error(y_test, xgb_pred)
np.sqrt(mean_squared_error(y_test, xgb_pred))

lgb= LGBMRegressor(objective='regression')
lgb.fit(X_train, y_train)
lgb_pred = lgb.predict(X_test)
accuracy = lgb.score(X_test,y_test)
'Accuracy: ' + str(np.round(accuracy*100, 2)) + '%'
mean_absolute_error(y_test, lgb_pred)
mean_squared_error(y_test, lgb_pred)
np.sqrt(mean_squared_error(y_test, lgb_pred))

from yellowbrick.regressor import ResidualsPlot
visualizer = ResidualsPlot(xgb)
visualizer.fit(X_train, y_train)  
visualizer.score(X_test, y_test)
visualizer

rid_pred_t=rid.predict(X_train)
la_pred_t=la.predict(X_train)
plt.scatter(la_pred_t, y_train, c = "blue", marker = "s", label = "Training data")
plt.scatter(la_pred, y_test, c = "lightgreen", marker = "s", label = "Validation data")
plt.title("Lasso regression")
plt.xlabel("Predicted values")
plt.ylabel("Real values")
plt.legend(loc = "upper left")
plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")
plt.show()