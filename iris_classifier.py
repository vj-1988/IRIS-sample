# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 11:26:21 2018

@author: Vijay anand
"""

import pandas as pd
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import pickle

#==============================================================================
# Main
#==============================================================================

def main():
    
    clf = SVC()
    le = LabelEncoder()
    cols=['sepal_length','sepal_width','petal_length','petal_width']
    df=pd.read_csv('iris.csv')
    
    feats=df[cols].values
    labels=list(df['species'].values)
    
    newLabels=np.reshape(le.fit_transform(labels),(150,1))
    
    
    
    model=clf.fit(feats,newLabels)
    
    pickle.dump(model, open('iris_svm.pkl', 'wb'))
    
    new_df=pd.DataFrame(np.hstack((feats,newLabels)),columns=cols+['labels'])
    
    
    new_df.to_csv('iris_processed.csv',index=None)
    
    
if __name__=='__main__':
    main()