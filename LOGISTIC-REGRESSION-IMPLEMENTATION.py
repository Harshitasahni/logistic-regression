#!/usr/bin/env python
# coding: utf-8

# In[21]:


#IMPORTING THE PACKAGES 
import csv
import pandas as pd
import numpy as np
import itertools
from scipy.sparse import csr_matrix
from scipy.sparse import save_npz, load_npz
from pandas import read_csv
from scipy.sparse import hstack
from math import exp
from sklearn import preprocessing


# In[22]:


#Preparing the training data

#Reading the training data
data=pd.read_csv('training.csv',header=None)

#Dropping the first Column-id
new_X_train = data.drop([0],axis=1)

#Saving the classes associated with each document
data_class  = new_X_train.iloc[:,61189] 

#Dropping the last column-class
new_X_train = new_X_train.drop(new_X_traindata.columns[-1],axis=1)

#Adding 1 as the first column in all 12000 documents
A=np.ones(12000)
new_X_train.insert(0,column=0,value=A)

#Saving the testing data as a csr matrix.
sparse_mtx_train=csr_matrix(new_X_train)
save_npz('testing_sparse',sparse_mtx_test)
sparse_mtx=load_npz('training_sparse_new.npz')


# In[23]:


#storing the size of training data
m=sparse_mtx.shape[0]
n=sparse_mtx.shape[1]


# In[24]:


#Saving class associated with each document in seprate csv file
df = pd.DataFrame({ "class" :data_class })
df.to_csv(r'C:\Users\harsh\Desktop\dataclass.csv', index=False)
data_class=pd.read_csv('dataclass.csv',header=None)
data_class.shape


# In[25]:


#Weights function initializes the weights to 0 
def weights():
    w=np.zeros((20,61189))
    w[:,:]=0.00
    #CONVERTING INTO A CSR MATRIX
    w=csr_matrix(w)
    return w


# In[26]:


#Probabilty function calculates the probality of each document associated with each class
def probX(W,sparse_mtx):
    #Dot product of W and training data 
    probarr=W.dot(sparse_mtx.T) 
    #sigmoid function
    probarr=probarr.expm1()
    #normalizing 
    probarr=norml(probarr) 
    #return the resultant probabilty 
    return(probarr)


# In[27]:


#Normalizing the result using sklearn preprocessing function
def norml(mtx):
    X_normalized = preprocessing.normalize(mtx, norm='l2',axis=0)
    return X_normalized


# In[139]:


# Delta matrix will store 1 in the index of the actual class for each document 
def delta(y, unique_classes):
    output = np.zeros((len(unique_classes), y.shape[0]),dtype=np.float16)
    for c in unique_classes:
        output[c-1, :] = np.where(y==c, 1., 0.).transpose()      
    #returning the csr_matrix     
    return csr_matrix(output)


# In[147]:


# Main trianing function where we change the values for learning rate penalty and number of iteartions.
def training(X):     
    #normalizing 
    x = norml(X)
    #calling the weight function
    w=weights() 
    #change the values for learing rate and penalty here 
    learning_rate=.0075
    penalty=.001
    iterations=500
    #calling the delta function
    u=np.unique(data_class)
    d=csr_matrix(delta(data_class,u))   
    for i in range(iterations):
        h = probX(w, x)
        #Calculating the error
        error = (d - h)
        #Making the predictions
        prediction = error * x
        #Updating the weights
        w += learning_rate * (prediction - penalty * w)       
    return w 


# In[148]:


# calling training function 
# weight will store the final weights evaluated from the training process
weight=training(sparse_mtx)


# #testing data 
# 

# In[40]:


#reading the testing data
testdata=pd.read_csv(r'C:\Users\harsh\Desktop\testing.csv',header=None)


# In[41]:


#saving the documents id before dropping it
documents=testdata.iloc[:,0]  


# In[14]:


#dropping the id column
data_new_test=testdata.drop([0],axis=1)


# In[15]:


#Adding 1 for the entire first column
A=np.ones(6774)
data_new_test.insert(0,column=0,value=A)


# In[16]:


#Making test data set as a sparse matrix 
sparse_mtx_test=csr_matrix(data_new_test)


# In[17]:


#Saving it as file (for easy loading)
save_npz('testing_sparse',sparse_mtx_test)


# In[102]:


#Loading the data file
sparse_test=load_npz('testing_sparse.npz')


# In[103]:


#shape of test dataset 
m=sparse_test.shape[0]
n=sparse_test.shape[1]


# In[149]:


#calculate prob.associated with each document and later finding the maximum among all which will be our predicted class.
def testfunction(Xtest,weight):
    Xtest=norml(Xtest)
    A=probX(weight,Xtest)
    #col_argmax will contain the class to which document has the highest probability 
    col_argmax = [A.getcol(i).A.argmax() for i in range(A.shape[1])]    
    return col_argmax    


# In[150]:


#Calling the test function and saving the predicted class in coltest
coltest=testfunction(sparse_test,weight)  


# In[151]:


#add 1 to class predicted since our evaluation starts from 0- 19 and real classes are from 1-20
for i in range(6774):
    coltest[i]=coltest[i]+1


# In[152]:


#Saving into csv file for calculating the accuracies through kaggle submittion
df = pd.DataFrame({"id" : documents, "class" : coltest})
df.to_csv(r'C:\Users\harsh\Desktop\iris.csv', index=False)


# In[153]:


#predicted classes
coltest 


# In[ ]:




