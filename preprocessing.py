# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 15:28:44 2019

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
import datetime
import nltk
from nltk.stem import WordNetLemmatizer

#get rid of '_'
def underscore(df,dtype):
    if dtype==1:
        new_columns=[]
        for col in df.columns:
            new_columns.append(re.sub('_','',col))
        df.columns=new_columns
    else:
        for col in df.columns:
            df[col]=df[col].str.replace('_',' ')
    return df
#lower all string
def lower(df,dtype):
    if dtype==1:
        new_columns=[]
        for col in df.columns:
            new_columns.append(col.lower())
        df.columns=new_columns
    else:
        for col in df.columns:
            df[col]=df[col].str.lower()
    return df


def rstring_to_num(ser,spe_c,num_type):
    ser=ser.str.rstrip(spe_c).astype(num_type)
    return ser

def str_dt_num(ser,dt_format,dtype):
    if dtype==1:
        ser=ser.apply(lambda x:datetime.datetime.strptime(x,dt_format))
    else:
        ser=ser.str[:10].apply(lambda x:datetime.datetime.strptime(x,dt_format))
    ser=(datetime.datetime(2015,12,31)-ser).dt.days/30
    return ser

def emplength_num(ser,dtype):
    if dtype==1:
        ser.replace('n/a',np.nan,inplace=True)
        ser.replace('<1 year','0',inplace=True)
        ser.replace(to_replace='[^0-9]+',value='',inplace=True,regex=True)
    else:
        ser=ser/12
    ser.fillna(value=-999,inplace=True)
    ser=ser.astype(int)
    return ser
    
def desc_num(ser):
    ser[ser.notnull()]=1
    ser[ser.isnull()]=0
    ser=ser.astype(int)
    return ser

#Lematizer 
def Lemmatizer(ser):
    ser=ser.str.lower()
    ser=ser.fillna('no description')
    wordnet_lemmatizer=WordNetLemmatizer()
    for i,title in enumerate(ser):
        if title:
            wordlist=title.split(' ')
            newlist=[]
            for word in wordlist:
                if word!='':
                    word=wordnet_lemmatizer.lemmatize(word,pos='n')
                    newlist.append(word)
            ser.iloc[i]=' '.join(newlist)
    return ser


#Frequency encoding: emptitle, addrzip, addrstate
def frequency_encoding(df,col):
    freqdf=df.groupby(col).size().reset_index()
    freqdf.columns=[col,col+'_freq']
    res=pd.merge(df,freqdf,how='left',on=col)
    return res


#Ordinal feature encoding: grade, subgrade
def ordinal_encoding(df,ser1,ser2):
    Dic_grade = {"A": 1,
                 "B": 2,
                 "C": 3,
                 "D": 4,
                 "E": 5,
                 "F": 6,
                 "G": 7
                 }
    df[ser1]=df[ser1].map(Dic_grade)
    df[ser2]=df[ser2].apply(lambda x: (Dic_grade[x[0]] - 1) * 5 + int(x[1]))
    return df

#Examine the feature
def feature_separation(df,cols):
    all_null_feature=[]
    num_feature=[]
    ob_feature=[]
    for col in cols:
        if df[col].isnull().sum()==df.shape[0]:
            all_null_feature.append(col)
        else:
            if df[col].dtype == 'object':
                ob_feature.append(col)
            else:
                num_feature.append(col)
    return all_null_feature, num_feature, ob_feature