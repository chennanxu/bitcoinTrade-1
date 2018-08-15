#encoding:utf-8
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error

from app import HuobiService
from helpler import ts_to_time


def get_model():
    kline_data = HuobiService.get_kline('btcusdt','1min',2000)
    kline_data = kline_data['data']

    train_df = pd.DataFrame(kline_data)

    train_df['oc'] = (train_df['open'] - train_df['close'])
    train_df['lh'] = (train_df['high'] - train_df['low'])
    label = train_df[:-1]['close']

    train_df = train_df[1:]
    train_df = train_df.reset_index(drop=True)
    print(train_df)
    train_df['label'] = label

    select_feat = ['open', 'high', 'low', 'close', 'vol', 'amount', 'count', 'oc', 'lh']
    label_df = pd.DataFrame(index=train_df.index, columns=["label"])
    label_df["label"] = np.log(train_df["label"])
    train_df = train_df[select_feat]

    def rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

    regr = xgb.XGBRegressor(
        colsample_bytree=0.5,
        gamma=0.0,
        learning_rate=0.01,
        max_depth=6,
        min_child_weight=1.5,
        n_estimators=2200,
        reg_alpha=0.9,
        reg_lambda=0.6,
        subsample=0.8,
        seed=2018,
        silent=1)
    regr.fit(train_df, label_df)
    curkline = HuobiService.get_kline('btcusdt', '1min', 1)

    testdata = pd.DataFrame(curkline['data'])
    testdata['oc'] = (testdata['open'] - testdata['close'])
    testdata['lh'] = (testdata['high'] - testdata['low'])

    testdata = testdata[select_feat]

    # train_data = xgb.DMatrix(train_data)
    # print(train_data)
    y_pred_xgb = regr.predict(testdata)
    y_pred = np.exp(y_pred_xgb)
    print(y_pred)
    return regr

get_model()
