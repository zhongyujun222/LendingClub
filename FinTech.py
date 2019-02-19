# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 15:19:04 2019

@author: Yujun
"""

import pandas as pd
import numpy as np
import re
import os
import pickle
from sklearn.externals import joblib
import json
import requests
pd.options.display.max_columns=None

import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import nltk
from nltk.stem import WordNetLemmatizer

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from bayes_opt import BayesianOptimization

path=r"E:\Yujun\DataAppLab\FinTech"
os.chdir(path)
import preprocessing as pr
import modeling as md


#%% 
#Load data from local
history1=pd.read_csv(os.path.join(path,'LoanStats3d.csv'),index_col=0)
current1=pd.read_csv(os.path.join(path,'current_loans.csv'),index_col=0)
history1=history1.reindex(sorted(history1.columns),axis=1)
dollar_df1=pd.read_csv(os.path.join(path,'dollar_df.csv'),index_col=0)

#%%
#if long=1, means it will go thtough a step that lemmantizer words which will take a long time. If bypass, set long=0
#if save=1, means it will output the results to a local directory
def preprocessing(history,current,dollar_df,long,save):
    history=pr.underscore(history,1)
    current=pr.lower(current,1)
    #Add new column to separate train and test
    history1['train_flag']=1
    current1['train_flag']=0
    #Manually map the historical column names to current column names
    history=history.rename(index=str,columns={'zipcode':'addrzip','loanamnt':'loanamount',
                                  'fundedamnt':'fundedamount',
                                  'verificationstatus':'isincv',
                                  'verificationstatusjoint':'isincvjoint',
                                  'numacctsever120pd':'numacctsever120ppd'})
    #Extract the common column names from history and current
    common_columns=list(set(history.columns)&set(current.columns))
    common_columns.extend(['loanstatus'])  
    #Only keep data with common columns
    history=history[common_columns]
    current['loanstatus']=np.NAN
    current=current[common_columns]
    #Convert some datatype in history
    history.intrate=pr.rstring_to_num(history.intrate,'%','float')
    history.revolutil=pr.rstring_to_num(history.revolutil,'%','float')
    history.earliestcrline=pr.str_dt_num(history.earliestcrline,'%b-%y',1)
    history.emplength=pr.emplength_num(history.emplength,1)
    history.term=history.term.str[:3].astype('int')
    #Conver some datatype in current
    current.earliestcrline=pr.str_dt_num(current.earliestcrline,'%Y-%m-%d',0)
    current.emplength=pr.emplength_num(current.emplength,0)
    #Only select 3 yr loans for history and current. Only select fully paid or charged off loans
    history=history[(history.term==36)&((history.loanstatus=='Fully Paid')|(history.loanstatus=='Charged Off'))]
    history['loanstatus']=history.loanstatus.map({'Fully Paid':0,'Charged Off':1})
    current=current[(current.term==36)]
    #Combine history and current together. The next conversions will be done in both history and current
    history=history.reset_index().drop(columns='id')
    current=current.reset_index().drop(columns='index')
    total=pd.concat([history,current],axis=0)
    #convert the data from lower_col into lower_keys; get rid of '_' in data from underscore_col
    lower_col=['applicationtype','disbursementmethod','initialliststatus','isincv','isincvjoint']
    underscore_col=['isincv','isincvjoint']
    total[lower_col]=pr.lower(total[lower_col],0)
    total[underscore_col]=pr.underscore(total[underscore_col],0)
    #Convert description into numeric: with description:1; no description:0
    total.desc=pr.desc_num(total.desc)
    #Convert the first three numbers in zipcode to numeric
    total.addrzip=pr.rstring_to_num(total.addrzip,'x','int')
    #If not time stringent, lemmatizer emptitle
    if long==1:
        total.emptitle=pr.Lemmatizer(total.emptitle)
    #unify some spelling
    total.emptitle=total.emptitle.str.replace('tecnician','technitian')
    total.emptitle=total.emptitle.str.replace('registered ','')
    #frequency encoding 'emptitle', 'addrzip', 'addrstate' 
    total=pr.frequency_encoding(total,'emptitle')
    total=pr.frequency_encoding(total,'addrzip')
    total=pr.frequency_encoding(total,'addrstate')
    #ordinal encoding 'grade', 'subgrade'
    total=pr.ordinal_encoding(total,'grade','subgrade')
    #Feature engineer to create a new feature: the absolute remained money after debt paid
    total['remain_income_abs']=total['annualinc']*(100-total['dti'])/100    
    total=pd.merge(total,dollar_df,how='left',on='addrstate')
    #Applied adjustment of dollar value to 'annualinc' and 'remain_income_abs'
    total['annualinc']=total['annualinc']*total['dollar_value']
    total['remain_income_abs']=total['remain_income_abs']*total['dollar_value']
    #If an output is needed
    if save==1:
        total.to_csv(os.path.join(path,'total_bf_one_hot_encoding.csv'))
    #separate features into three types: numerical, categorical and all_null
    all_null_feature_h, num_feature_h, ob_feature_h=pr.feature_separation(total,total.columns)
    #Remove 'addrstate', 'emptitle', and 'secappearliestcrline' from categorical feature list
    ob_feature_h.remove('addrstate')
    ob_feature_h.remove('emptitle')
    ob_feature_h.remove('secappearliestcrline')
    #If length matches, one hot encoding the rest categorical features
    if (len(ob_feature_h)+len(num_feature_h)+4)==len(total.columns):
        total=pd.concat([total[num_feature_h],pd.get_dummies(total[ob_feature_h])],axis=1)
        if save==1:
            total.to_csv(os.path.join(path,'total_one_hot_encoded.csv'))
    #sort the column names
    total=total.reindex(sorted(total.columns),axis=1)
    return total


total=preprocessing(history1,current1,dollar_df1,0,0)
total.head()


#%%

def xgb_evaluate(eta,
                 min_child_weight,
                 colsample_bytree,
                 max_depth,
                 subsample,
                 gamma
                 ):
    xgtrain = xgb.DMatrix(X_train, y_train, missing = np.NAN)
    params = dict()
    params['objective'] = 'binary:logistic'
    params['eta'] = eta
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
#Model Building
#opt=1:use bayesian optimization
#n_iter:number of boosting rounds
#n_imp:number of important features    
def XGBmodel(total,opt,n_iter,n_imp):
    dtrain,dvalid,dtest,X_train,y_train,X_valid,y_valid,X_test,y_test=md.data_split(total)
    test_params = {"objective": "binary:logistic",
          "booster" : "gbtree",
          "eta": 0.05,
          "max_depth": 6,
          "subsample": 0.632,
          "colsample_bytree": 0.7,
          "silent": 1,
          "seed": 1234,
          "eval_metric": "auc",
          "min_child_weight": 5}
    model,eval_results,bst_ntree,df_importance = md.model_init(test_params,dtrain,dvalid,n_iter)
    md.draw_ROC(model, dtrain, dvalid, dtest, y_train, y_valid, y_test)
    md.draw_fea_imp(df_importance,n_imp)
    if opt==1:
        tune_params = {'eta':(0.01,0.05),
                      'max_depth': (4, 6),
                      'min_child_weight': (0, 20),
                      'colsample_bytree': (0.2, 0.8),
                      'subsample': (0.5, 1),
                      'gamma': (0, 2)
                     }
        model,bst_params,score=md.Optimizer(xgb_evaluate,tune_params)
        bst_params['params']['max_depth']=np.int(bst_params['params']['max_depth'])
        bst_params['params']['min_child_weight']=np.int(bst_params['params']['min_child_weight'])
        model,eval_results,bst_ntree,df_importance = md.model_init(bst_params['params'],dtrain,dvalid,n_iter)
    else:
        bst_params=test_params
    return model,bst_params,eval_results,bst_ntree,df_importance
    
model,bst_params,eval_results,bst_ntree,df_importance=XGBmodel(total,0,1500,20)
 

   
        
def save(model,pickle_name,dat_name):
    with open(pickle_name,'wb') as pkl_file:
        pickle.dump(model,pkl_file)
    joblib.dump(model,dat_name)
    
save(model,'test.pkl','test.joblib.dat')    




