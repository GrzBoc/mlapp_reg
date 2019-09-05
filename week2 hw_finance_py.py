


# https://github.com/ranaroussi/yfinance
# install yfinance
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import yfinance as yf
from pandas_datareader import data as pdr
import ta

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2, f_regression,mutual_info_regression

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


d_start='2016-01-01'
d_end='2019-08-30'

apple_hst=pdr.get_data_yahoo('AAPL', start=d_start, end=d_end)
msft_hst=pdr.get_data_yahoo('MSFT', start=d_start, end=d_end)
dupont_hst=pdr.get_data_yahoo('DD', start=d_start, end=d_end)
alcoa_hst=pdr.get_data_yahoo('AA', start=d_start, end=d_end)
# ... or df = pdr.DataReader("AAPL", 'yahoo', start, end) # drugi argument definuje źródło informacji
# https://pandas-datareader.readthedocs.io/en/latest/remote_data.html


# library for technical analysis indicators
# https://github.com/bukosabino/ta
# install ta 
# other indicators like hma can be found e.g.:
# https://github.com/backtrader/backtrader/tree/master/backtrader/indicators


dd=alcoa_hst.copy()


dd['target_1']=dd['Adj Close'].shift(-1)/dd['Adj Close']-1
dd['target_3']=dd['Adj Close'].shift(-3)/dd['Adj Close']-1
dd['target_7']=dd['Adj Close'].shift(-7)/dd['Adj Close']-1
dd['target_14']=dd['Adj Close'].shift(-14)/dd['Adj Close']-1
dd['target_30']=dd['Adj Close'].shift(-30)/dd['Adj Close']-1

dd['s00_return_1']=dd.Close/dd.Close.shift(1)-1
dd['s00_return_3']=dd.Close/dd.Close.shift(3)-1
dd['s03_return_1']=dd.s00_return_1.shift(3)
dd['s03_return_3']=dd.s00_return_3.shift(3)
dd['s05_return_1']=dd.s00_return_1.shift(5)
dd['s05_return_3']=dd.s00_return_3.shift(5)


dd['s00_rsi_14']=ta.rsi(dd.Close, 14)
dd['s00_rsi_7']=ta.rsi(dd.Close, 7)
dd['s00_willR_14']=ta.wr(dd.High,dd.Low,dd.Close,14)
dd['s00_willR_7']=ta.wr(dd.High,dd.Low,dd.Close,7)
dd['s00_stoch_sig_14_3']=ta.stoch_signal(dd.High,dd.Low,dd.Close,14,3)
dd['s00_stoch_sig_7_3']=ta.stoch_signal(dd.High,dd.Low,dd.Close,7,3)
dd['s00_cci_20_0015']=ta.cci(dd.High,dd.Low,dd.Close,20,0.015)
dd['s00_cci_20_005']=ta.cci(dd.High,dd.Low,dd.Close,20,0.05)
dd['s00_macd_12_26_9']=ta.macd_diff(dd.Close,12,26,9)
dd['s00_macd_7_14_9']=ta.macd_diff(dd.Close,7,14, 9)
dd['s00_kst_9']=ta.kst(dd.Close)-ta.kst_sig(dd.Close)

dd['s01_rsi_14']=dd.s00_rsi_14.shift(+1)
dd['s01_rsi_7']=dd.s00_rsi_7.shift(+1)
dd['s01_willR_14']=dd.s00_willR_14.shift(+1)
dd['s01_willR_7']=dd.s00_willR_7.shift(+1)
dd['s01_stoch_sig_14_3']=dd.s00_stoch_sig_14_3.shift(+1)
dd['s01_stoch_sig_7_3']=dd.s00_stoch_sig_7_3.shift(+1)
dd['s01_cci_20_0015']=dd.s00_cci_20_0015.shift(+1)
dd['s01_cci_20_005']=dd.s00_cci_20_005.shift(+1)
dd['s01_macd_12_26_9']=dd.s00_macd_12_26_9.shift(+1)
dd['s01_macd_7_14_9']=dd.s00_macd_7_14_9.shift(+1)
dd['s01_kst_9']=dd.s00_kst_9.shift(+1)

dd['s03_rsi_14']=dd.s00_rsi_14.shift(+3)
dd['s03_rsi_7']=dd.s00_rsi_7.shift(+3)
dd['s03_willR_14']=dd.s00_willR_14.shift(+3)
dd['s03_willR_7']=dd.s00_willR_7.shift(+3)
dd['s03_stoch_sig_14_3']=dd.s00_stoch_sig_14_3.shift(+3)
dd['s03_stoch_sig_7_3']=dd.s00_stoch_sig_7_3.shift(+3)
dd['s03_cci_20_0015']=dd.s00_cci_20_0015.shift(+3)
dd['s03_cci_20_005']=dd.s00_cci_20_005.shift(+3)
dd['s03_macd_12_26_9']=dd.s00_macd_12_26_9.shift(+3)
dd['s03_macd_7_14_9']=dd.s00_macd_7_14_9.shift(+3)
dd['s03_kst_9']=dd.s00_kst_9.shift(+3)

dd['s05_rsi_14']=dd.s00_rsi_14.shift(+5)
dd['s05_rsi_7']=dd.s00_rsi_7.shift(+5)
dd['s05_willR_14']=dd.s00_willR_14.shift(+5)
dd['s05_willR_7']=dd.s00_willR_7.shift(+5)
dd['s05_stoch_sig_14_3']=dd.s00_stoch_sig_14_3.shift(+5)
dd['s05_stoch_sig_7_3']=dd.s00_stoch_sig_7_3.shift(+5)
dd['s05_cci_20_0015']=dd.s00_cci_20_0015.shift(+5)
dd['s05_cci_20_005']=dd.s00_cci_20_005.shift(+5)
dd['s05_macd_12_26_9']=dd.s00_macd_12_26_9.shift(+5)
dd['s05_macd_7_14_9']=dd.s00_macd_7_14_9.shift(+5)
dd['s05_kst_9']=dd.s00_kst_9.shift(+5)

dd['s10_rsi_14']=dd.s00_rsi_14.shift(+10)
dd['s10_rsi_7']=dd.s00_rsi_7.shift(+10)
dd['s10_willR_14']=dd.s00_willR_14.shift(+10)
dd['s10_willR_7']=dd.s00_willR_7.shift(+10)
dd['s10_stoch_sig_14_3']=dd.s00_stoch_sig_14_3.shift(+10)
dd['s10_stoch_sig_7_3']=dd.s00_stoch_sig_7_3.shift(+10)
dd['s10_cci_20_0015']=dd.s00_cci_20_0015.shift(+10)
dd['s10_cci_20_005']=dd.s00_cci_20_005.shift(+10)
dd['s10_macd_12_26_9']=dd.s00_macd_12_26_9.shift(+10)
dd['s10_macd_7_14_9']=dd.s00_macd_7_14_9.shift(+10)
dd['s10_kst_9']=dd.s00_kst_9.shift(+10)



trgt_1=dd.iloc[60:(dd.shape[0]-30),dd.columns.get_loc('target_1')]
trgt_3=dd.iloc[60:(dd.shape[0]-30),dd.columns.get_loc('target_3')]
trgt_7=dd.iloc[60:(dd.shape[0]-30),dd.columns.get_loc('target_7')]
trgt_14=dd.iloc[60:(dd.shape[0]-30),dd.columns.get_loc('target_14')]
trgt_30=dd.iloc[60:(dd.shape[0]-30),dd.columns.get_loc('target_30')]

X=dd.iloc[60:(dd.shape[0]-30),dd.columns.get_loc('target_30')+1:]


X_train=X.iloc[60:(X.shape[0]-100),:]
Y_train=trgt_3.iloc[60:(X.shape[0]-100)]

X_test=X.iloc[(X.shape[0]-100):X.shape[0],:]
Y_test=trgt_3.iloc[(X.shape[0]-100):X.shape[0]]



# application of standard scaler
sc = StandardScaler()
X_train = pd.DataFrame(sc.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(sc.transform (X_test), columns=X_test.columns)

selector=SelectKBest(score_func=f_regression,k=30)
#selector=SelectKBest(score_func=mutual_info_regression,k=2)

selector.fit_transform(X_train,Y_train)
cols = selector.get_support(indices=True)
X_train_adj=X_train.iloc[:,cols]
X_test_adj=X_test.iloc[:,cols]


#linear regression
linreg = LinearRegression(normalize=True)
linreg.fit(X_train_adj,Y_train)
Y_train_pred_linreg = linreg.predict(X_train_adj)
Y_test_pred_linreg = linreg.predict(X_test_adj)

#Quadratic regression
quadreg = make_pipeline(PolynomialFeatures(2), Ridge())
quadreg.fit(X_train_adj,Y_train)
Y_train_pred_quadreg = quadreg.predict(X_train_adj)
Y_test_pred_quadreg = quadreg.predict(X_test_adj)

#Polynomial regression
polnreg = make_pipeline(PolynomialFeatures(3), Ridge())
polnreg.fit(X_train_adj,Y_train)
Y_train_pred_polnreg = polnreg.predict(X_train_adj)
Y_test_pred_polnreg = polnreg.predict(X_test_adj)


# Ridge regression (optimize alpha)
ridgereg = Ridge(alpha=0.3,normalize=True)
ridgereg.fit(X_train_adj,Y_train)
Y_train_pred_ridgereg = ridgereg.predict(X_train_adj)
Y_test_pred_ridgereg = ridgereg.predict(X_test_adj)

# Lasso regression
lassoreg = Lasso(alpha=0.4,normalize=True, max_iter=1e6)
lassoreg.fit(X_train_adj,Y_train)
Y_train_pred_lassoreg = lassoreg.predict(X_train_adj)
Y_test_pred_lassoreg = lassoreg.predict(X_test_adj)

# Elastic Net regression

elasticreg = ElasticNet(alpha=0.5,l1_ratio=0.7)
elasticreg.fit(X_train_adj,Y_train)
Y_train_pred_elasticreg = elasticreg.predict(X_train_adj)
Y_test_pred_elasticreg = elasticreg.predict(X_test_adj)

# KNN regressor

knnreg = KNeighborsRegressor(n_neighbors=2)
knnreg.fit(X_train_adj,Y_train)
Y_train_pred_knnreg = knnreg.predict(X_train_adj)
Y_test_pred_knnreg = knnreg.predict(X_test_adj)



print('MSE')
print("Linear_______: train %.4f  test %.4f" % (mean_squared_error(Y_train, Y_train_pred_linreg), mean_squared_error(Y_test, Y_test_pred_linreg)))
print("Quadratic____: train %.4f  test %.4f" % (mean_squared_error(Y_train, Y_train_pred_quadreg),mean_squared_error(Y_test, Y_test_pred_quadreg)))
print("Polynomial___: train %.4f  test %.4f" % (mean_squared_error(Y_train, Y_train_pred_polnreg),mean_squared_error(Y_test, Y_test_pred_polnreg)))
print("Ridge________: train %.4f  test %.4f" % (mean_squared_error(Y_train, Y_train_pred_ridgereg),mean_squared_error(Y_test, Y_test_pred_ridgereg)))
print("Lasso________: train %.4f  test %.4f" % (mean_squared_error(Y_train, Y_train_pred_lassoreg),mean_squared_error(Y_test, Y_test_pred_lassoreg)))
print("Elastic Net__: train %.4f  test %.4f" % (mean_squared_error(Y_train, Y_train_pred_elasticreg),mean_squared_error(Y_test, Y_test_pred_elasticreg)))
print("KNN regresor_: train %.4f  test %.4f" % (mean_squared_error(Y_train, Y_train_pred_knnreg),mean_squared_error(Y_test, Y_test_pred_knnreg)))

print('MAE')
print("Linear_______: train %.4f  test %.4f" % (mean_absolute_error(Y_train, Y_train_pred_linreg), mean_absolute_error(Y_test, Y_test_pred_linreg)))
print("Quadratic____: train %.4f  test %.4f" % (mean_absolute_error(Y_train, Y_train_pred_quadreg),mean_absolute_error(Y_test, Y_test_pred_quadreg)))
print("QPolynomial__: train %.4f  test %.4f" % (mean_absolute_error(Y_train, Y_train_pred_polnreg),mean_absolute_error(Y_test, Y_test_pred_polnreg)))
print("Ridge________: train %.4f  test %.4f" % (mean_absolute_error(Y_train, Y_train_pred_ridgereg),mean_absolute_error(Y_test, Y_test_pred_ridgereg)))
print("Lasso________: train %.4f  test %.4f" % (mean_absolute_error(Y_train, Y_train_pred_lassoreg),mean_absolute_error(Y_test, Y_test_pred_lassoreg)))
print("Elastic Net__: train %.4f  test %.4f" % (mean_absolute_error(Y_train, Y_train_pred_elasticreg),mean_absolute_error(Y_test, Y_test_pred_elasticreg)))
print("KNN regresor_: train %.4f  test %.4f" % (mean_absolute_error(Y_train, Y_train_pred_knnreg),mean_absolute_error(Y_test, Y_test_pred_knnreg)))

print('r2_score')
print("Linear_______: %.6f" % (r2_score(Y_train, Y_train_pred_linreg)))
print("Quadratic____: %.6f" % (r2_score(Y_train, Y_train_pred_quadreg)))
print("Polynomial___: %.6f" % (r2_score(Y_train, Y_train_pred_polnreg)))
print("Ridge________: %.6f" % (r2_score(Y_train, Y_train_pred_ridgereg)))
print("Lasso________: %.6f" % (r2_score(Y_train, Y_train_pred_lassoreg)))
print("Elastic Net__: %.6f" % (r2_score(Y_train, Y_train_pred_elasticreg)))
print("KNN regresor_: %.6f" % (r2_score(Y_train, Y_train_pred_knnreg)))


x=np.arange(1,Y_test.shape[0]+1)

plt.subplot(4, 2, 1)
plt.plot(x,Y_test, '-', lw=2, color='r')
plt.plot(x,Y_test_pred_linreg, '-', lw=2)
plt.title('Linear Reg - Returns,test vs pred')
plt.grid(True)
plt.subplot(4, 2, 2)
plt.plot(x,abs(Y_test-Y_test_pred_linreg), '-', lw=2,color='k')
plt.title('Absolute Errors')

plt.subplot(4, 2, 3)
plt.plot(x,Y_test, '-', lw=2, color='r')
plt.plot(x,Y_test_pred_quadreg, '-', lw=2)
plt.title('Quadratic Reg - Returns,test vs pred')
plt.grid(True)
plt.subplot(4, 2, 4)
plt.plot(x,abs(Y_test-Y_test_pred_quadreg), '-', lw=2,color='k')
plt.title('Absolute Errors')

plt.subplot(4, 2, 5)
plt.plot(x,Y_test, '-', lw=2, color='r')
plt.plot(x,Y_test_pred_quadreg, '-', lw=2)
plt.title('Polynomial Reg - Returns,test vs pred')
plt.grid(True)
plt.subplot(4, 2, 6)
plt.plot(x,abs(Y_test-Y_test_pred_quadreg), '-', lw=2,color='k')
plt.title('Absolute Errors')

plt.subplot(4, 2, 7)
plt.plot(x,Y_test, '-', lw=2, color='r')
plt.plot(x,Y_test_pred_knnreg, '-', lw=2)
plt.title('KNN Reg - Returns,test vs pred')
plt.grid(True)
plt.subplot(4, 2, 8)
plt.plot(x,abs(Y_test-Y_test_pred_knnreg), '-', lw=2,color='k')
plt.title('Absolute Errors')


plt.tight_layout()
plt.show()


















