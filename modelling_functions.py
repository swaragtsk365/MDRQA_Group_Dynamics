import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.model_selection import RepeatedStratifiedKFold
import random
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
import sklearn
import pickle
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from itertools import compress, product
from itertools import combinations
from sklearn.preprocessing import normalize

def Standardize(data):
   '''
   This is a function to standardize the data
   that is mean centering and scaling by standard deviation
   '''
   Mu=np.mean(data,axis=0, keepdims=True)
   Sigma=np.std(data,axis=0, keepdims=True)
   return (data-Mu)/Sigma


def Subsets(List):
  '''
  This is a function for creating superset of a list
  '''
  N=len(List)
  combs=[]
  for i in range(N):
    subset=list(combinations(List,i+1))
    #print(subset)
    SS=[]
    for j in range(len(subset)):
      SS_s=[]
      for k in range(len(subset[j])):
        SS_s.append(subset[j][k])
        
      print(SS_s)
      combs.append(SS_s)


  return combs

def ROC_bootstrapping_KNN(data,k, variables,algorithm,metric,n_splits,n_repeats):
  '''
  This function computes bootstrapped ROC values with error bars for a given dataset and given list of features
  data: data in pandas data frame
  algorithm: algorithm used in knn
  metric: distance metric used in knn
  n_split: number of split to be made in the repeated stratified k fold cross validation
  n_repeats: number of repeats for repeated stratified k fold cross validation
  variables: list of variables to be used
  '''
  #variables=['recc_rate','percent_det','avg_diag','percent_lam','diag_ent','vert_max']
  Matrix=[]
  print('Starting Bootstrapping.....')
  for x in variables:
    Matrix.append(np.array(data[x]))
    
  

  MAT=np.array(Matrix)
  y=np.array(data['outcome'])

  Y=y
  MAT=Standardize(MAT.T)
  TPR=[]
  FPR=[]
  THRESH=[]
  TRUTHS=[]
  YTRUTH=[]
  YTRAIN=[]
  BIN_TEST=[]
  BIN_TRAIN=[]
  YTEST=[]
  ACC_SCORE=[]
  LLOSS=[]
  
  rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,random_state=36851234)
  for train_index, test_index in tqdm(rskf.split(MAT, Y)):
      forest = KNeighborsClassifier(n_neighbors=k, algorithm='brute',metric='l2')
      X_train=MAT[train_index]
      Y_train=Y[train_index]
      X_test=MAT[test_index]
      Y_test=Y[test_index]
      
      
      	
      
      forest = KNeighborsClassifier(n_neighbors=k, algorithm='brute',metric='l2')
      forest.fit(X_train,Y_train)
      
      Y_pred_train=forest.predict_proba(X_train)[:, 1]
      Y_pred_test=forest.predict_proba(X_test)[:, 1]
      print('--------------------------------------------------------------------------------------------------------------------------------------')
      print('TEST:',accuracy_score(Y_test,forest.predict(X_test)))
      print('TRAIN:',accuracy_score(Y_train,forest.predict(X_train)))
      print('Probs:',Y_pred_test)
      YTRAIN.append([Y_train,Y_pred_train])
      YTEST.append([Y_test,Y_pred_test])
      BIN_TEST.append([Y_test, forest.predict(X_test)])

      BIN_TRAIN.append([Y_train,forest.predict(X_train)])
      ACC_SCORE.append(accuracy_score(Y_test, forest.predict(X_test)))
      LLOSS.append(log_loss(Y_test, Y_pred_test))
      print('log loss:',log_loss(Y_test, Y_pred_test))

      
      try:
         fpr, tpr, thresholds=sklearn.metrics.roc_curve(Y_test,Y_pred_test)
         print('fpr:',fpr)
         print('tpr:',tpr)
         print('threshold:',thresholds)
         
         TPR.append(tpr)
         FPR.append(fpr)
         THRESH.append(thresholds)
      except ValueError:
         print('error')
         pass
         
         
      print('--------------------------------------------------------------------------------------------------------------------------------------')
    
  BS={'TPR':TPR,'FPR':FPR,'Threshold':THRESH,'TRAIN':YTRAIN,'TEST':YTEST,'BIN_TRAIN':BIN_TRAIN,'BIN_TEST':BIN_TEST}
  print('finished bootstrapping....')

  return BS, TRUTHS, ACC_SCORE, LLOSS

def All_subset_selection_KNN(data,k, variables,algorithm,metric,n_splits,n_repeats, outfile):
  '''
  function to do the best subset selection for KNN
  data: input data(pandas dataframe)
  k   : the number of neighbours to be used in the KNN algorithm
  variables: list of variables from which the best subset selection should be done
  algorithm: algorithm used in knn
  metric: distance metric used in knn
  n_split: number of split to be made in the repeated stratified k fold cross validation
  n_repeats: number of repeats for repeated stratified k fold cross validation
  outfile: name of file to be used for writing the results
  
  '''
  subsets=Subsets(variables)
  model_infos={}
  FEATS=[]
  AVG_ACC=[]
  AVG_LLOSS=[]
  SD_ACC=[]
  SD_LLOSS=[]
  SN=[]
  i=0
  for elem in subsets:
    i=i+1
    BS, TRUTHS,ACC_SCORE,LLOSS=ROC_bootstrapping_KNN(data,k, elem,algorithm,metric,n_splits,n_repeats)
    FEATS.append(elem)
    AVG_ACC.append(np.mean(ACC_SCORE))
    AVG_LLOSS.append(np.mean(LLOSS))
    SD_ACC.append(np.std(ACC_SCORE))
    SD_LLOSS.append(np.std(LLOSS))
    SN.append(i)
    model_infos[i]={'BS':BS,'variables':TRUTHS}
    
  pickle_file=open(outfile+'.pkl','wb')
  pickle.dump(model_infos, pickle_file)
  pickle_file.close()

  DICT={'SN':SN,'features':FEATS,'avg_accuracy':AVG_ACC,'avg_logloss':AVG_LLOSS,'sd_accuracy':SD_ACC,'sd_logloss':SD_LLOSS}
  data_out=pd.DataFrame.from_dict(DICT)
  data_out.to_csv(outfile+'.csv')
  
def ROC_bootstrapping_QDA(data, variables,n_splits,n_repeats):
  '''
  This function computes bootstrapped ROC values with error bars for a given dataset and given list of features
  data: data in pandas data frame
  n_split: number of split to be made in the repeated stratified k fold cross validation
  n_repeats: number of repeats for repeated stratified k fold cross validation
  variables: list of variables to be used
  '''
  #variables=['recc_rate','percent_det','avg_diag','percent_lam','diag_ent','vert_max']
  Matrix=[]
  print('Starting Bootstrapping.....')
  for x in variables:
    Matrix.append(np.array(data[x]))
    
  

  MAT=np.array(Matrix)
  y=np.array(data['outcome'])

  Y=y
  MAT=normalize(MAT.T)
  TPR=[]
  FPR=[]
  THRESH=[]
  TRUTHS=[]
  YTRUTH=[]
  YTRAIN=[]
  BIN_TEST=[]
  BIN_TRAIN=[]
  YTEST=[]
  ACC_SCORE=[]
  LLOSS=[]
  
  rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,random_state=36851234)
  for train_index, test_index in tqdm(rskf.split(MAT, Y)):
      forest = QuadraticDiscriminantAnalysis()
      X_train=MAT[train_index]
      Y_train=Y[train_index]
      X_test=MAT[test_index]
      Y_test=Y[test_index]
      
      
      	
      
      forest = QuadraticDiscriminantAnalysis()
      forest.fit(X_train,Y_train)
      
      Y_pred_train=forest.predict_proba(X_train)[:, 1]
      Y_pred_test=forest.predict_proba(X_test)[:, 1]
      print('--------------------------------------------------------------------------------------------------------------------------------------')
      print('TEST:',accuracy_score(Y_test,forest.predict(X_test)))
      print('TRAIN:',accuracy_score(Y_train,forest.predict(X_train)))
      print('Probs:',Y_pred_test)
      YTRAIN.append([Y_train,Y_pred_train])
      YTEST.append([Y_test,Y_pred_test])
      BIN_TEST.append([Y_test, forest.predict(X_test)])

      BIN_TRAIN.append([Y_train,forest.predict(X_train)])
      ACC_SCORE.append(accuracy_score(Y_test, forest.predict(X_test)))
      #LLOSS.append(log_loss(Y_test, Y_pred_test))
      #print('log loss:',log_loss(Y_test, Y_pred_test))

      
      try:
         fpr, tpr, thresholds=sklearn.metrics.roc_curve(Y_test,Y_pred_test)
         print('fpr:',fpr)
         print('tpr:',tpr)
         print('threshold:',thresholds)
         
         TPR.append(tpr)
         FPR.append(fpr)
         THRESH.append(thresholds)
      except ValueError:
         print('error')
         pass
         
         
      print('--------------------------------------------------------------------------------------------------------------------------------------')
    
  BS={'TPR':TPR,'FPR':FPR,'Threshold':THRESH,'TRAIN':YTRAIN,'TEST':YTEST,'BIN_TRAIN':BIN_TRAIN,'BIN_TEST':BIN_TEST}
  print('finished bootstrapping....')

  return BS, TRUTHS, ACC_SCORE, LLOSS

def All_subset_selection_QDA(data,metric,n_splits,n_repeats,outfile):
  '''
  function to do the best subset selection for QDA
  data: input data(pandas dataframe)
  k   : the number of neighbours to be used in the KNN algorithm
  variables: list of variables from which the best subset selection should be done
  n_split: number of split to be made in the repeated stratified k fold cross validation
  n_repeats: number of repeats for repeated stratified k fold cross validation
  outfile: name of file to be used for writing the results
  
  '''
  subsets=Subsets(variables)
  model_infos={}
  FEATS=[]
  AVG_ACC=[]
  AVG_LLOSS=[]
  SD_ACC=[]
  SD_LLOSS=[]
  SN=[]
  i=0
  for elem in subsets:
    i=i+1
    BS, TRUTHS,ACC_SCORE,LLOSS=ROC_bootstrapping_QDA(data,n_splits,n_repeats)
    FEATS.append(elem)
    AVG_ACC.append(np.mean(ACC_SCORE))
    #AVG_LLOSS.append(np.mean(LLOSS))
    SD_ACC.append(np.std(ACC_SCORE))
    SD_LLOSS.append(np.std(LLOSS))
    SN.append(i)
    model_infos[i]={'BS':BS,'variables':TRUTHS}
    
  pickle_file=open(outfile+'.pkl','wb')
  pickle.dump(model_infos, pickle_file)
  pickle_file.close()

  DICT={'SN':SN,'features':FEATS,'avg_accuracy':AVG_ACC,'sd_accuracy':SD_ACC,'sd_logloss':SD_LLOSS}
  data_out=pd.DataFrame.from_dict(DICT)
  data_out.to_csv(outfile+'.csv')
  
def K_selector(data, subsets,algorithm,metric,n_splits,n_repeats,outfile):
   '''
   This function will generate data about log loss and accuracy for different k values for KNN from which we can choose the appropriate k value
   data: data(pandas dataframe)
   subsets: set of all variables used for the classifier
   algorithm: algorithm used for KNN
   metric: metric used in KNN
   n_splits: number of splits made in repeated stratified K- fold cross validation
   n_repeats: number of repeats in the repeated stratified K fold cross validation
   outfile: name of file to save output
   '''
  
  model_infos={}
  FEATS=[]
  AVG_ACC=[]
  AVG_LLOSS=[]
  SD_ACC=[]
  SD_LLOSS=[]
  SN=[]
  i=0
  for i in range(1,16):
    BS, TRUTHS,ACC_SCORE,LLOSS=ROC_bootstrapping4(data,i, subsets,algorithm,metric,n_splits,n_repeats)
    FEATS.append(i)
    AVG_ACC.append(np.mean(ACC_SCORE))
    AVG_LLOSS.append(np.mean(LLOSS))
    SD_ACC.append(np.std(ACC_SCORE))
    SD_LLOSS.append(np.std(LLOSS))
    SN.append(i)
    model_infos[i]={'BS':BS,'variables':TRUTHS}
    
  pickle_file=open(outfile+'.pkl','wb')
  pickle.dump(model_infos, pickle_file)
  pickle_file.close()

  DICT={'SN':SN,'K':FEATS,'avg_accuracy':AVG_ACC,'avg_logloss':AVG_LLOSS,'sd_accuracy':SD_ACC,'sd_logloss':SD_LLOSS}
  data_out=pd.DataFrame.from_dict(DICT)
  data_out.to_csv(outfile+'.csv')


def Best_subset_KNN(data, subset,algorithm,metric,n_splits,n_repeats,outfile):
  '''
  This is a function to do the repeated stratified k-fold cv on selected variables
  data: input data(pandas dataframe)
  subset: subset of variables chosen
  algorithm: algorithm for KNN
  metric: metric for KNN
  n_splits: number of splits for repeated stratified k-fold cross validation
  n_repeats: number of repeats for repeated stratified k-fold cross validation
  outfile: name of file to save the output
  '''
  
  model_infos={}
  FEATS=[]
  AVG_ACC=[]
  AVG_LLOSS=[]
  SD_ACC=[]
  SD_LLOSS=[]
  SN=[]
  i=0
  for i in range(8,9):
    BS, TRUTHS,ACC_SCORE,LLOSS=ROC_bootstrapping_KNN(data,i, subsets,algorithm,metric,n_splits,n_repeats)
    FEATS.append(i)
    AVG_ACC.append(np.mean(ACC_SCORE))
    AVG_LLOSS.append(np.mean(LLOSS))
    SD_ACC.append(np.std(ACC_SCORE))
    SD_LLOSS.append(np.std(LLOSS))
    SN.append(i)
    model_infos[i]={'BS':BS,'variables':TRUTHS}
    
  pickle_file=open(outfile+'.pkl','wb')
  pickle.dump(model_infos, pickle_file)
  pickle_file.close()

  DICT={'SN':SN,'K':FEATS,'avg_accuracy':AVG_ACC,'avg_logloss':AVG_LLOSS,'sd_accuracy':SD_ACC,'sd_logloss':SD_LLOSS}
  data_out=pd.DataFrame.from_dict(DICT)
  data_out.to_csv(outfile+'.csv')
  
def Best_subset_QDA(data, subset,n_splits,n_repeats,outfile):
  '''
  This is a function to do the repeated stratified k-fold cv on selected variables
  data: input data(pandas dataframe)
  subset: subset of variables chosen
  n_splits: number of splits for repeated stratified k-fold cross validation
  n_repeats: number of repeats for repeated stratified k-fold cross validation
  outfile: name of file to save the output
  '''
  
  model_infos={}
  FEATS=[]
  AVG_ACC=[]
  AVG_LLOSS=[]
  SD_ACC=[]
  SD_LLOSS=[]
  SN=[]
  i=0
  for i in range(8,9):
    BS, TRUTHS,ACC_SCORE,LLOSS=ROC_bootstrapping_QDA(data,i, subsets,n_splits,n_repeats)
    FEATS.append(i)
    AVG_ACC.append(np.mean(ACC_SCORE))
    AVG_LLOSS.append(np.mean(LLOSS))
    SD_ACC.append(np.std(ACC_SCORE))
    SD_LLOSS.append(np.std(LLOSS))
    SN.append(i)
    model_infos[i]={'BS':BS,'variables':TRUTHS}
    
  pickle_file=open(outfile+'.pkl','wb')
  pickle.dump(model_infos, pickle_file)
  pickle_file.close()

  DICT={'SN':SN,'K':FEATS,'avg_accuracy':AVG_ACC,'avg_logloss':AVG_LLOSS,'sd_accuracy':SD_ACC,'sd_logloss':SD_LLOSS}
  data_out=pd.DataFrame.from_dict(DICT)
  data_out.to_csv(outfile+'.csv')


