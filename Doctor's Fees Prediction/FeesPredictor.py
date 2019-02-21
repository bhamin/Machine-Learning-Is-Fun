
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np


# In[ ]:

#======================================= Data Overview ============================================


# In[2]:

train = pd.read_excel('Final_Train.xlsx', sheet_name=None)
train.head()

test = pd.read_excel('Final_Test.xlsx', sheet_name=None)
test.head()


# In[3]:

sub = pd.read_excel('Sample_submission.xlsx', sheet_name=None)
sub.head()


# In[4]:

def PercentNan(data):
    data_nan = (data.isnull().sum() / data.shape[0]) * 100
    print(data_nan)

def Summarize(data):
    print(data.shape)
    print(data.dtypes)
    PercentNan(data)
    
print("Train")
Summarize(train)
print("Test")
Summarize(test)


# In[6]:

#========================================== Coding Section ==========================================


# In[7]:

import gc
import numpy as np
import pandas as pd
import random

#qualification location fees    
def qlfLocPowerGetter(row, qlfLocFee):
    q = row['Qualification']
    l = row['Locality']
    row['qlfLocPower'] = int((row['PowerQualificationScore'] + row['profLocPower'])/2)
    if (q, l) in qlfLocFee.keys():
        row['qlfLocPower'] = int(qlfLocFee[q][l])
        
    return row

def profLocPowerGetter(row, profLocFee):
    p = row['Profile']
    l = row['Locality']
    row['profLocPower'] = int((row['PowerProfileScore'] + row['PowerLocationScore'])/2)
    if (p, l) in profLocFee.keys():
        row['profLocPower'] = int(profLocFee[p][l])
        
    return row

def getPowerLocation(df):
    #get unique degs
    uniqueQlf = (df.Locality.unique())
    
    dictMapQlfMeanFee = {}
    for i in range(len(uniqueQlf)):
        dictMapQlfMeanFee.update({uniqueQlf[i]: {'sum':0,'cnt':0,'mean':0}})
        
    for idx, row in df.iterrows():
        d = row['Locality']
        fee = row['Fees']
        
        if pd.isnull(fee):
            continue
        
        dictMapQlfMeanFee[d]['sum'] = dictMapQlfMeanFee[d]['sum'] + fee
        dictMapQlfMeanFee[d]['cnt'] = dictMapQlfMeanFee[d]['cnt'] + 1

    zeroMeanCnt = 0
    for l in uniqueQlf:
        s = dictMapQlfMeanFee[l]['sum']
        c = dictMapQlfMeanFee[l]['cnt']
        
        if l in dictMapQlfMeanFee.keys():
            if c == 0:
                dictMapQlfMeanFee[l]['mean'] = 200
                zeroMeanCnt = zeroMeanCnt + 1
            else:
                dictMapQlfMeanFee[l]['mean'] = int(s/c)
        else:
            print(">>> ", l)
          
    print("loc with zero fees entry(count):", zeroMeanCnt, " -- as new location found in test set")
    
    df['PowerLocationScore'] = df.Locality.map(lambda l: (int)(dictMapQlfMeanFee[l]['mean']))
    
def getPowerProfile(df):
    uniqueQlf = (df.Profile.unique())
    
    dictMapQlfMeanFee = {}
    for i in range(len(uniqueQlf)):
        dictMapQlfMeanFee.update({uniqueQlf[i]: {'sum':0,'cnt':0,'mean':0}})
        
    for idx, row in df.iterrows():
        d = row['Profile']
        fee = row['Fees']
        
        if pd.isnull(fee):
            continue
        
        dictMapQlfMeanFee[d]['sum'] = dictMapQlfMeanFee[d]['sum'] + fee
        dictMapQlfMeanFee[d]['cnt'] = dictMapQlfMeanFee[d]['cnt'] + 1

    zeroMeanCnt = 0
    for p in uniqueQlf:
        s = dictMapQlfMeanFee[p]['sum']
        c = dictMapQlfMeanFee[p]['cnt']
        
        if p in dictMapQlfMeanFee.keys():
            if c == 0:
                dictMapQlfMeanFee[p]['mean'] = 200
                zeroMeanCnt = zeroMeanCnt + 1
            else:
                dictMapQlfMeanFee[p]['mean'] = int(s/c)
        else:
            print(">>> ", p)
          
    print("Profile with zero fees entry(count):", zeroMeanCnt, " -- as new profile found in test set")
    
    df['PowerProfileScore'] = df.Profile.map(lambda p: (int)(dictMapQlfMeanFee[p]['mean']))
    

#qualification Experience fees
def qlfExpPowerGetter(row, qlfExpFee):
    q = row['Qualification']
    e = row['Experience']
    row['qlfExpPower'] = row['PowerQualificationScore']
    if (q, e) in qlfExpFee.keys():
        row['qlfExpPower'] = int(qlfExpFee[q][e])
        
    return row

def getPowerQualification(df):
    
    uniqueQlf = (df.Qualification.unique())
    
    dictMapQlfMeanFee = {}
    for i in range(len(uniqueQlf)):
        dictMapQlfMeanFee.update({uniqueQlf[i]: {'sum':0,'cnt':0,'mean':0}})
        
    for idx, row in df.iterrows():
        d = row['Qualification']
        fee = row['Fees']
        
        if pd.isnull(fee):
            continue
        
        dictMapQlfMeanFee[d]['sum'] = dictMapQlfMeanFee[d]['sum'] + fee
        dictMapQlfMeanFee[d]['cnt'] = dictMapQlfMeanFee[d]['cnt'] + 1

    zeroMeanCnt = 0
    for qlf in uniqueQlf:
        s = dictMapQlfMeanFee[qlf]['sum']
        c = dictMapQlfMeanFee[qlf]['cnt']
        
        if qlf in dictMapQlfMeanFee.keys():
            if c == 0:
                dictMapQlfMeanFee[qlf]['mean'] = 200
                zeroMeanCnt = zeroMeanCnt + 1
            else:
                dictMapQlfMeanFee[qlf]['mean'] = int(s/c)
        else:
            print(">>> ", qlf)
          
    print("Qlf with zero fees entry(count):", zeroMeanCnt, " -- as new qlf found in test set")
    
    df['PowerQualificationScore'] = df.Qualification.map(lambda q: (int)(dictMapQlfMeanFee[q]['mean']))

#get degs fees
def getPowerScore(df):
    uniqueQlf = (df.Qualification.unique())
    uniqueDeg = set([])
    
    for qlf in uniqueQlf:
        degs = [d.strip() for d in qlf.split(',')]
        for deg in degs:
            uniqueDeg.add(deg)
    
    uniqueDeg = list(uniqueDeg)
    
    dictMapDegMeanFee = {}
    for i in range(len(uniqueDeg)):
        dictMapDegMeanFee.update({uniqueDeg[i]: {'sum':0,'cnt':0,'mean':0}})
        
    for idx, row in df.iterrows():
        degs = row['Degs']
        fee = row['Fees']
        
        if pd.isnull(fee):
            continue
        
        for d in degs:
            dictMapDegMeanFee[d]['sum'] = dictMapDegMeanFee[d]['sum'] + fee
            dictMapDegMeanFee[d]['cnt'] = dictMapDegMeanFee[d]['cnt'] + 1

    zeroMeanCnt = 0
    for deg in uniqueDeg:
        s = dictMapDegMeanFee[deg]['sum']
        c = dictMapDegMeanFee[deg]['cnt']
        
        if deg in dictMapDegMeanFee.keys():
            if c == 0:
                dictMapDegMeanFee[deg]['mean'] = 200
                zeroMeanCnt = zeroMeanCnt + 1
            else:
                dictMapDegMeanFee[deg]['mean'] = int(s/c)
        else:
            print(">>> ", deg)
          
    print("Deg with zero fees entry(count):", zeroMeanCnt, " -- as new degs found in test set")
    
    df['PowerScore'] = df.Degs.map(lambda degs: (int)(np.mean([dictMapDegMeanFee[d]['mean'] for d in degs])))
    
#split Place into city, Locality
def mapLocCity(row):
    vals = (row.Place.split(','))
    if len(vals) == 1:
        row['Locality'] = vals[0].strip()
        row['City'] = 'city'
    else:
        locality = ''
        for i in range(0,len(vals)-1):
            if locality != '':
                locality = locality + " "
            locality = locality + vals[i].strip()
        row['Locality'] = locality
        row['City'] = vals[len(vals)-1].strip()
    return row
    
def train_test():
    train_df = pd.read_excel('Final_Train.xlsx', sheet_name=None)
    test_df = pd.read_excel('Final_Test.xlsx', sheet_name=None)
    
    #combine train test 
    test['Fees'] = np.nan
    df = train_df.append(test_df)
    del train_df, test_df
    gc.collect()
    
    #>> MISC: delete misc info
    del df['Miscellaneous_Info']
    gc.collect()
    
    #>> RATINGS: handle null Ratings(~50%) - 1.remove % | 2.convert string to int | 3.fill nulls with mean(+-)rand_std 
    df.Rating = df.Rating.map(lambda r: np.nan if pd.isnull(r) else (int)(r.replace('%','')))
    ratings_mean = df.Rating.mean()
    rating_std = df.Rating.std()
    print(ratings_mean, df.Rating.std())
    df.Rating = df.Rating.map(lambda r: int(random.uniform(ratings_mean - rating_std, ratings_mean + rating_std))-2 if pd.isnull(r) else r)
    df.Rating.astype(int)
    print(df.Rating.mean(), df.Rating.std())
    
    #>> EXPERIENCE: 
    df.Experience = df.Experience.map(lambda e: int(e.split(' ')[0]))
    df.Experience.astype(int)
    df.loc[ df['Experience'] <= 5, 'Experience'] = 1
    df.loc[(df['Experience'] > 5) & (df['Experience'] <= 10), 'Experience'] = 2
    df.loc[(df['Experience'] > 10) & (df['Experience'] <= 15), 'Experience'] = 3
    df.loc[(df['Experience'] > 15) & (df['Experience'] <= 30), 'Experience'] = 4
    df.loc[(df['Experience'] > 30), 'Experience'] = 5
    
    #>> PLACES: handle null palces(~0.5%) - fill null as 'city, india'
    print(df.Place.isnull().sum())
    df['Place'] = df.Place.map(lambda p: 'locality, city' if pd.isnull(p) else p)
    print(df.Place.isnull().sum())
    
    #new feature: city, locality from place
    df['Locality'] = df.Place
    df['City'] = df.Place 
    df = df.apply(mapLocCity, axis='columns')
    del df['Place']
    gc.collect()
    
    dictMapCity = {}
    lstCity = df.City.unique()
    for i in range(len(lstCity)):
        dictMapCity.update({lstCity[i]:i})
    
    dictMapLocality = {}
    lstLocality = df.Locality.unique()
    for i in range(len(lstLocality)):
        dictMapLocality.update({lstLocality[i]:i})
        
    df['City'] = df.City.map(dictMapCity)
    df['Locality'] = df.Locality.map(dictMapLocality)
    
    #>> PROFILE:
    dictMapProfile = {}
    lstProfile = df.Profile.unique()
    for i in range(len(lstProfile)):
        dictMapProfile.update({lstProfile[i]:i})
        
    df['Profile'] = df.Profile.map(dictMapProfile)
    
    #>> Power Score : Degs(based on degrees owned) + QUALIFICATION(based on full qualification)
    df['Degs'] = df.Qualification.map(lambda qlf: [d.strip() for d in qlf.split(',')])
    getPowerScore(df)
    getPowerQualification(df)
    
    #>> QLF_EXP - mean fees power(if not avail QLF_EXP then use QualificationPower) 
    #             + del PowerQualificationScore(as redundant feature)
    tr = df[df.Fees.notnull()]
    qlfExpFee = tr.groupby(['Qualification','Experience']).Fees.agg('mean')
    qlfExpFee.reset_index()
    df = df.apply(qlfExpPowerGetter, args = (qlfExpFee,), axis = "columns")
    #print(df.head(2))
    
    #>> Locality Qualification Power
    getPowerLocation(df)
    getPowerProfile(df)
    
    tr = df[df.Fees.notnull()]
    profLocFee = tr.groupby(['Profile','Locality']).Fees.agg('mean')
    profLocFee.reset_index()
    df = df.apply(profLocPowerGetter, args = (profLocFee,), axis = "columns")
    
    tr = df[df.Fees.notnull()]
    qlfLocFee = tr.groupby(['Qualification','Locality']).Fees.agg('mean')
    qlfLocFee.reset_index()
    df = df.apply(qlfLocPowerGetter, args = (qlfLocFee,), axis = "columns")
    #print(df.head(2))
    
    del df['PowerProfileScore']
    del df['PowerLocationScore']
    del df['PowerQualificationScore']    
    
    #delete extra colms
    del df['Qualification']
    del df['Degs']
    
    gc.collect()
    
    print(df.head(2))
    
    #***** Understand Data ******
    #print(df.head(2))
    #print(df.describe())
    #print("Total Samples:", len(df))
    #print("Cities:", len(df.City.unique()))
    #print("Locs:", len(df.Locality.unique()))
    #print("Profiles:", len(df.Profile.unique()))
    #print("Experiences:", len(df.Experience.unique()), ">>>", df.Experience.min(), "to", max(list(df.Experience.unique())))
    #print(df.Experience.value_counts())
    #print(df.PowerCategory.value_counts())
    #print(df.isnull().any())
    
    return df


# In[9]:

from contextlib import contextmanager
import time
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from pandas import ExcelWriter

import matplotlib.pyplot as plt

from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus


# to track execution timing of block of code
@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

def doTrainPredict(train_df, test_df):
    
    #Traing Data
    y = train_df['Fees']
    del train_df['Fees']
    X = train_df
    
    #Test Data
    X_test = test_df
    
    #cross-validation
    
    gbrc = GradientBoostingRegressor(n_estimators = 1000, learning_rate = 0.01, random_state = 0, loss="ls")
    rmse = np.sqrt([-x for x in (cross_val_score(gbrc, X, y, cv=8,scoring='neg_mean_squared_error'))])
    print("RMSE(GBR)",rmse, "\n mean",np.mean(rmse), "std",np.std(rmse))
    
    
    rfc = RandomForestRegressor(n_estimators = 1000, random_state = 42)    rmse = np.sqrt([-x for x in (cross_val_score(rfc, X, y, cv=8,scoring='neg_mean_squared_error'))])
    print("RMSE(RF)",rmse, "\n mean",np.mean(rmse), "std",np.std(rmse))
    
    #----- Models
    gbr = GradientBoostingRegressor(n_estimators = 1000, learning_rate = 0.01, random_state = 0, loss="ls")
    rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
    
    #---- Training
    print("Training started - RF Reg...")
    gbr.fit(X,y)
    rf.fit(X,y)
    print("Training over...")
    
    #---- Predict
    
    predGBR = gbr.predict(X_test)
    for i in range(len(predGBR)):
        predGBR[i] = int(predGBR[i])
    print("GBR:",predGBR[0:20])
    
    predRF = rf.predict(X_test)
    for i in range(len(predRF)):
        predRF[i] = int(predRF[i])
    print("RF:",predRF[0:20])
    
    predMix = [val for val in predRF]
    for i in range(len(predMix)):
        predMix[i] = int(0.35*predRF[i] + 0.65*predGBR[i])
    print("Mix:",predMix[0:20])
    
    #tmp: model of train predictions
    '''
    TP = rf.predict(X)
    T = y
    sumSquared = 0
    plainErr = []
    for i in range(len(TP)):
        err = TP[i] - T[i]
        sumSquared = sumSquared + (err**2)
        if err < 0:
            plainErr.append(-err)
        else:
            plainErr.append(err)
        
    MSE = (sumSquared / len(TP))
    RMSE = np.sqrt(int(MSE))
    print("Training predictions: MSE:", MSE, "RMSE", RMSE, " | >> Plain: mean", np.mean(plainErr), "std", np.std(plainErr)
         , "min", np.min(plainErr), "max", np.max(plainErr))
    '''
    
    #----- submission
    
    submissionGBR = pd.DataFrame({
                "Fees": predGBR
            })
    writerGBR = ExcelWriter('fees_prediction_GBR_v6.xlsx')
    submissionGBR.to_excel(writerGBR,'Sheet1', index = False)
    writerGBR.save()
    
    submissionRF = pd.DataFrame({
                "Fees": predRF
            })
    #submissionRF.to_csv('fees_prediction_RF_v3.csv', mode = 'w', index = False)
    writerRF = ExcelWriter('fees_prediction_RF_v6.xlsx')
    submissionRF.to_excel(writerRF,'Sheet1', index = False)
    writerRF.save()
    
    submissionMix = pd.DataFrame({
                "Fees": predMix
            })
    writerMix = ExcelWriter('fees_prediction_RF_GBR_v6.xlsx')
    submissionMix.to_excel(writerMix,'Sheet1', index = False)
    writerMix.save()
    
    #------- visulaize feature imp
    features = X.columns.values
    
    importances = gbr.feature_importances_
    indices = np.argsort(importances)

    plt.title('Feature Importances: GBR')
    plt.barh(range(len(indices)), importances[indices], color='#8f63f4', align='center')
    plt.yticks(range(len(indices)), features[indices])
    plt.xlabel('Relative Importance')
    plt.show()
    
    importances = rf.feature_importances_
    indices = np.argsort(importances)

    plt.title('Feature Importances: RF')
    plt.barh(range(len(indices)), importances[indices], color='#8f63f4', align='center')
    plt.yticks(range(len(indices)), features[indices])
    plt.xlabel('Relative Importance')
    plt.show()
    
def main(debug = False):
    with timer("Process train & test"):
        df = train_test()
        train_df = df[df['Fees'].notnull()]
        test_df = df[df['Fees'].isnull()]
        del test_df['Fees']
        train_df.to_csv("train_feature.csv", index = False)
        test_df.to_csv("test_feature.csv", index = False)
    with timer("Training & Perdict:"):
        doTrainPredict(train_df, test_df)
        
if __name__ == "__main__":
    with timer("Full model run"):
        main(debug=False)


# In[ ]:




# In[ ]:



