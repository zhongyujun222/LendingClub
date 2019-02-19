# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 16:04:24 2019

@author: zhongy2
"""
import pandas as pd
import numpy as np
import re
import os
import pickle
import json
import requests
pd.options.display.max_columns=None

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from bayes_opt import BayesianOptimization

def data_split(total):
    X=total[total.train_flag==1].drop(columns='loanstatus')
    y=total[total.train_flag==1]['loanstatus']
    X_test=total[total.train_flag==0].drop(columns='loanstatus')
    y_test=total[total.train_flag==0]['loanstatus']
    X_train,X_valid,y_train,y_valid=train_test_split(X,y,test_size=0.2,random_state=2014)
    dtrain = xgb.DMatrix(X_train, y_train, missing = np.NAN)
    dvalid = xgb.DMatrix(X_valid, y_valid, missing = np.NAN)
    dtest = xgb.DMatrix(X_test, y_test, missing = np.NAN)
    return dtrain,dvalid,dtest,X_train,y_train,X_valid,y_valid,X_test,y_test

def model_init(params,dtrain,dvalid,n):
    #Initiate a xgboost model
    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    num_boost_round = n
    evals_results={}
    gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist,\
                    early_stopping_rounds= 50,evals_result=(evals_results))
    evals_results=pd.DataFrame.from_dict(evals_results,orient='index')
    bst_ntree=gbm.best_ntree_limit
    importance = gbm.get_fscore()
    df_importance = pd.DataFrame.from_dict(importance,orient='index')
    df_importance=df_importance.reset_index()
    df_importance.columns=['feature', 'fscore']
    df_importance['fscore'] = df_importance['fscore'] / df_importance['fscore'].sum()
    df_importance.sort_values(['fscore'], ascending=False, inplace=True)
    return gbm,evals_results,bst_ntree,df_importance

def draw_ROC(model, dtrain, dvalid, dtest, y_train, y_valid, y_test ):
    probas_ = model.predict(dvalid, ntree_limit=model.best_ntree_limit)
    probas_1 = model.predict(dtrain, ntree_limit=model.best_ntree_limit)
#    probas_2 = model.predict(dtest, ntree_limit=model.best_ntree_limit)
    fpr, tpr, thresholds = roc_curve(y_valid, probas_)
    fpr_1, tpr_1, thresholds_1 = roc_curve(y_train, probas_1)
#    fpr_2, tpr_2, thresholds_2 = roc_curve(y_test, probas_2)
    roc_auc = auc(fpr, tpr)
    roc_auc_1 = auc(fpr_1, tpr_1)
#    roc_auc_2 = auc(fpr_2, tpr_2)
    print("Area under the ROC curve - validation: %f" % roc_auc)
    print("Area under the ROC curve - train: %f" % roc_auc_1)
#    print("Area under the ROC curve - test: %f" % roc_auc_2)
    # Plot ROC curve
    plt.figure(figsize=(8,8))
    plt.plot(fpr, tpr, label='ROC curve - test(AUC = %0.2f)' % roc_auc, color='r')
    plt.plot(fpr_1, tpr_1, label='ROC curve - train (AUC = %0.2f)' % roc_auc_1, color='b')
#    plt.plot(fpr_2, tpr_2, label='ROC curve - train (AUC = %0.2f)' % roc_auc_2, color='g')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for lead score model')
    plt.legend(loc="lower right")
    plt.show()



def draw_fea_imp(df,n):
    #​plt.figure()
    df[:n].plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
    plt.title('XGBoost Feature Importance')
    plt.xlabel('relative importance')
    plt.show()


#Model Tuning


def xgb_evaluate(min_child_weight,
                 colsample_bytree,
                 max_depth,
                 subsample,
                 gamma
                 ):
    xgtrain = xgb.DMatrix(X_train, y_train, missing = np.NAN)
    params = dict()
    params['objective'] = 'binary:logistic'
    params['eta'] = 0.05
    params['max_depth'] = int(max_depth )   
    params['min_child_weight'] = int(min_child_weight)
    params['colsample_bytree'] = colsample_bytree
    params['subsample'] = subsample
    params['gamma'] = gamma
    params['verbose_eval'] = False 
    cv_result = xgb.cv(params, xgtrain,
                       num_boost_round=100000,
                       nfold=3,
                       metrics={'auc'},
                       seed=1234,
                       callbacks=[xgb.callback.early_stop(50)])
    print(cv_result)
    return cv_result['test-auc-mean'].max()

def Optimizer(fun,params):
    xgb_BO = BayesianOptimization(fun, 
                                  params)
    xgb_BO.maximize(init_points=5, n_iter=40)
    xgb_BO_scores = pd.DataFrame([x['params'] for x in xgb_BO.res])
    xgb_BO_scores['score'] = pd.DataFrame([x['target'] for x in xgb_BO.res])
    xgb_BO_scores = xgb_BO_scores.sort_values(by='score',ascending=False)
    return xgb_BO, xgb_BO.max, xgb_BO_scores

