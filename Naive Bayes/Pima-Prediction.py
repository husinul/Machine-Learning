
# coding: utf-8

# ## Predicting Diabetes
# 
# 

# ## import libraries

# In[1]:

import pandas as pd               # pandas is a dataframe library
import matplotlib.pyplot as plt   # matplotlib plots data
import numpy as np                # numpy provides N-dim object support

# do plotting  inline instead of seperate window
get_ipython().magic('matplotlib inline')




# ## Load and Review data

# In[2]:

df = pd.read_csv("C:/Users/Arjun S Kumar/Downloads/demo/Notebooks/data/pima-data.csv")


# In[3]:

df.shape


# In[4]:

df.head(5)


# In[5]:

df.tail(5)
   


# In[6]:

df.isnull().values.any()


# In[7]:

df.corr()


# In[8]:

del df['skin']


# In[9]:

df.head(5)


# ## check Data type

# In[10]:

df.head(5)


# change true and false to 1 and 0

# In[11]:

diabetes_map = { True :1 , False : 0}
df['diabetes'] =df['diabetes'].map(diabetes_map)


# In[12]:

df.head(5)


#  ## check true /false info 

# In[13]:

num_true = len(df.loc[df['diabetes']== True])
num_false = len(df.loc[df['diabetes']== False])


# ## spliting the data as train and test data

# In[14]:

from sklearn.cross_validation import train_test_split
feature_col_names = ['num_preg','glucose_conc','diastolic_bp','thickness','insulin','bmi','diab_pred','age']
predicted_class_name = ['diabetes']
X = df[feature_col_names].values
y= df[predicted_class_name].values
split_test_size = 0.3
X_train , X_test, y_train , y_test  = train_test_split(X,y,test_size= split_test_size, random_state = 42)


# In[15]:

#plotting the correlation 
def plot_corr(df, size=10):
    """
    Function plots a graphical correlation matrix for each pair of columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot

    Displays:
        matrix of correlation between columns.  Blue-cyan-yellow-red-darkred => less to more correlated
                                                0 ------------------>  1
                                                Expect a darkred line running from top left to bottom right
    """

    corr = df.corr()    # data frame correlation function
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)   # color code the rectangles by correlation value
    plt.xticks(range(len(corr.columns)), corr.columns)  # draw x tick marks
    plt.yticks(range(len(corr.columns)), corr.columns)  # draw y tick marks
    
plot_corr(df)


# In[16]:

print("row missing insulin {0}".format(len(df.loc[df['insulin'] == 0])))


# ## we are imputing the missing value with mean

# In[17]:

from sklearn.preprocessing import Imputer
fill_0 = Imputer(missing_values=0, strategy="mean",axis=0) 

X_train = fill_0.fit_transform(X_train)
X_test = fill_0.fit_transform(X_test)


# ## Training Initial Algorithm - Naive Bayes

# In[18]:

from sklearn.naive_bayes import GaussianNB

nb_model = GaussianNB()

nb_model.fit(X_train, y_train.ravel())


# ## perfomance on Training data

# In[19]:

nb_predict_train = nb_model.predict(X_train)

from sklearn import metrics

print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_train,nb_predict_train)))


# ### Metrics
# 

# In[23]:


nb_predict_test =nb_model.predict(X_test)

print(" Confusion Matrix")
print("{0}".format(metrics.confusion_matrix(y_test, nb_predict_test, labels=[1,0])))
print("")

print("Classification Report")

print(metrics.classification_report(y_test, nb_predict_test ,labels=[1,0]))



# In[24]:

from sklearn.ensemble import RandomForestClassifier
 
rf_model = RandomForestClassifier(random_state = 5)
rf_model.fit (X_train, y_train.ravel())
    


# In[25]:

rf_predict_train= rf_model.predict(X_train)
print(" Accuracy : {0:.4f}".format(metrics.accuracy_score(y_train,rf_predict_train)))


# ## logistic regression

# In[29]:

from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression( C= 0.7,random_state= 5)
lr_model.fit(X_train, y_train.ravel())


# In[34]:

# setting regularization parameter 

C_start  = 0.1
C_end =5
C_inc =0.1
C_values, recall_scores = [],[]
C_val = C_start
best_recall_score = 0
while( C_val < C_end):
    C_values.append(C_val)
    lr_model_loop = LogisticRegression(C= C_val,class_weight ="balanced", random_state =5)
    lr_model_loop.fit(X_train, y_train.ravel())
    lr_predict_loop_test = lr_model_loop.predict(X_test)
    recall_score = metrics.recall_score(y_test, lr_predict_loop_test)
    recall_scores.append(recall_score)
    
    if(recall_score > best_recall_score):
        best_recall_score = recall_score
        best_lr_predict_test = lr_predict_loop_test
     
    C_val = C_val +C_inc
    
best_score_C_val = C_values[recall_scores.index(best_recall_score)]
print("1st max value of {0:.3f} occured at C= {1:.3f}".format(best_recall_score, best_score_C_val))
    
get_ipython().magic('matplotlib inline')
plt.plot(C_values, recall_scores, "-")
plt.xlabel("C value")
plt.ylabel("recall score")


# In[35]:

## logicstic regression cross validation ( Logicstic RegressionCV)

from sklearn.linear_model import LogisticRegressionCV
lr_cv_model = LogisticRegressionCV( n_jobs=-1, random_state= 5, Cs=3, cv= 10, refit =True ,class_weight ="balanced")
lr_cv_model.fit(X_train, y_train.ravel())


# In[ ]:



