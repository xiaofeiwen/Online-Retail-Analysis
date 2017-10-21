import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
df_train = pd.read_pickle('../data/train_transformed.p')
df_test = pd.read_pickle('../data/test_transformed.p')
order_products_compact = pd.read_hdf('../data/online_retail.h5','order_products_compact')

def f1_score(l_true,l_pred):
    tp = set(l_true).intersection(set(l_pred))
    if not len(tp):
        return 0
    fp = set(l_pred).difference(tp)
    fn = set(l_true).difference(tp)
    p = len(tp) / (len(tp) + len(fp))
    r = len(tp) / (len(tp) + len(fn))
    f1 = 2 * (p * r) / (p + r)
    return f1
def avg_f1_score(df,pred,order_products_compact=order_products_compact,thres=0.09):
    df_pred = pd.DataFrame({'order_id':df.order_id,'pred':pred,'product_id':df.product_id,
                            'prior_size_max':df.user_order_size_max,
                            'prior_size_mean':df.user_order_size_mean,
                            'prior_size_std':df.user_order_size_std})\
                .sort_values(['order_id','pred'],ascending = [True,False]).reset_index(drop=True)
    df_pred['pred_rank'] = df_pred.groupby('order_id').cumcount()
    df_pred['prior_size_2std'] = df_pred.prior_size_mean + df_pred.prior_size_std * 2
    df_pred = df_pred[df_pred.pred_rank < df_pred.prior_size_max]\
            .reset_index(drop=True)
    d = {}
    for row in df_pred.itertuples():
        order_id = row.order_id
        if row.pred_rank == 0 or row.pred > thres:
            try:
                d[order_id] += ' ' + str(row.product_id)
            except:
                d[order_id] = str(row.product_id)
    df_pred_compact = pd.DataFrame.from_dict(d, orient='index')

    df_pred_compact.reset_index(inplace=True)
    df_pred_compact.columns = ['order_id', 'y_pred']
    df_pred_compact['y_pred'] = df_pred_compact['y_pred'].str.split()
    df_pred_compact = df_pred_compact.merge(order_products_compact[['order_id','product_id']],how='left',
                                                       on='order_id')
    scores = []
    for row in df_pred_compact.itertuples():
        y_pred = row.y_pred
        y_true = row.product_id
        score = f1_score(y_true,y_pred)
        scores.append(score)
    df_pred_compact['f1_score'] = scores
    return np.mean(scores),df_pred_compact

f_to_use_tree = ['user_total_orders',
       'user_total_items', 'user_total_distinct_items',
       'user_average_days_between_orders', 'user_order_size_mean','user_order_size_max','user_order_size_std',
       'user_total_item_quantity', 'user_total_spent',
       'user_sum_days_between_orders', 'user_reorder_ratio',
       'order_hour_of_day', 'days_since_prior_order', 'days_since_ratio',
       'product_reorder_rate',
       'product_total_quantity_sold', 'product_avg_price', 'prod_first_buy',
       'prod_1reorder_ratio',
       'UP_orders', 'UP_orders_ratio', 'UP_total_quantity',
       'UP_order_rate_since_first_order']
f_to_use_lgr = ['user_total_orders', 'user_average_days_between_orders', 'user_order_size_mean', 
             'user_total_item_quantity', 'order_hour_of_day','order_dow', 'days_since_ratio',
             'product_orders', 'product_avg_price', 
             'UP_orders', 'UP_total_quantity', 
             'user_sum_days_between_orders','user_reorder_ratio','prod_1reorder_ratio']

# logistic regression
print ('training logistic regression...')
lgr = LogisticRegression(random_state=42,n_jobs=-1,C=100).fit(df_train[f_to_use_lgr],df_train['labels'].values)
test_pred_lgr = lgr.predict_proba(df_test[f_to_use_lgr])[:,1]

# LightGBM
print ('training lightGBM...')
d_train = lgb.Dataset(df_train[f_to_use_tree],label=df_train['labels'].values)
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss'},
    'num_leaves': 128,
    'max_depth': 8,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.95,
    'bagging_freq': 5,
    'learning_rate': 0.053,
}
bst = lgb.train(params, d_train, 100)
test_pred_lgb = bst.predict(df_test[f_to_use_tree])

# random forest
print ('training random forest..')
rfc = RandomForestClassifier(random_state = 42, n_estimators=100, max_depth = 7, n_jobs=-1,min_samples_split=100).\
        fit(df_train[f_to_use_tree],df_train['labels'].values)
test_pred_rf = rfc.predict_proba(df_test[f_to_use_tree])[:,1]

# XGBoost
print ('training XGBoost...')
d_train = xgb.DMatrix(df_train[f_to_use_tree],label=df_train['labels'].values)
xgb_params = {
    "objective"         : "reg:logistic"
    ,"eval_metric"      : "logloss"
    ,"eta"              : 0.15
    ,"max_depth"        : 8
    ,"min_child_weight" :10
    ,"gamma"            :0.70
    ,"subsample"        :0.76
    ,"colsample_bytree" :0.95
    ,"alpha"            :2e-05
    ,"lambda"           :10
}
bst = xgb.train(params=xgb_params, dtrain=d_train, num_boost_round=100)
test_pred_xgb = bst.predict(xgb.DMatrix(df_test[f_to_use_tree]))

# Model combination
test_pred = test_pred_lgb * 0.2 + test_pred_xgb * 0.5 + test_pred_rf * 0.1 + test_pred_lgr * 0.2

score,result = avg_f1_score(df_test,test_pred)
print ('test score is {}'.format(score))
result.columns = ['order_id','recommendations','ordered_products','f1_score']
result.set_index('order_id',inplace=True)
result.to_excel('../output/recommendations.xlsx')
print ('please see the output folder for recommendations')


