# datasets from import sklearn.databases from load_digits
# coding: utf-8

# In[1]:

from sklearn.datasets import load_digits
digits = load_digits()


# In[2]:

print type(digits)
print digits.data


# In[3]:

print digits.DESCR


# In[4]:

print digits.target


# In[5]:

print digits.target_names


# In[7]:

print digits.target.shape
print digits.data.shape


# In[11]:

X = digits.data
y = digits.target


# In[12]:

print X
print y


# In[15]:

from sklearn import svm
clf = svm.SVC(gamma=0.001, C=100)
clf.fit(X,y)
print('Prediction:'), clf.predict(digits.data[12])
print('Actual:'), y[12]


# In[ ]:



