#!/usr/bin/env python
# coding: utf-8

# ## HW3: Decision Tree, AdaBoost and Random Forest
# In hw3, you need to implement decision tree, adaboost and random forest by using only numpy, then train your implemented model by the provided dataset and test the performance with testing data
# 
# Please note that only **NUMPY** can be used to implement your model, you will get no points by simply calling sklearn.tree.DecisionTreeClassifier

# ## Load data
# The dataset is the Heart Disease Data Set from UCI Machine Learning Repository. See follow links for more information
# https://archive.ics.uci.edu/ml/datasets/heart+Disease

# In[342]:


import pandas as pd
import numpy as np
file_url = "http://storage.googleapis.com/download.tensorflow.org/data/heart.csv"
df = pd.read_csv(file_url)

train_idx = np.load('train_idx.npy')
test_idx = np.load('test_idx.npy')

train_df = df.iloc[train_idx]
test_df = df.iloc[test_idx]


# ## Question 1
# Gini Index or Entropy is often used for measuring the “best” splitting of the data. Please compute the Entropy and Gini Index of provided data. Please use the formula from [page 7 of hw3 slides](https://docs.google.com/presentation/d/1ish3jEr_6be0FK4kgOZa12nYAyJFh0P2LCNsNPOCiXo/edit#slide=id.g7703d1636d_0_21)

# In[5]:


def gini(sequence):
    return None

def entropy(sequence):
    return None


# In[14]:


# 1 = class 1,
# 2 = class 2
data = np.array([1,2,1,1,1,1,2,2,1,1,2])


# In[15]:


print("Gini of data is ", gini(data))


# In[16]:


print("Entropy of data is ", entropy(data))


# ## Question 2
# Implement the Decision Tree algorithm (CART, Classification and Regression Trees) and trained the model by the given arguments, and print the accuracy score on the test data. You should implement two arguments for the Decision Tree algorithm
# 1. **Criterion**: The function to measure the quality of a split. Your model should support “gini” for the Gini impurity and “entropy” for the information gain. 
# 2. **Max_depth**: The maximum depth of the tree. If Max_depth=None, then nodes are expanded until all leaves are pure. Max_depth=1 equals to split data once
# 

# In[7]:


class DecisionTree():
    def __init__(self, criterion='gini', max_depth=None):
        return None


# ### Question 2.1
# Using Criterion=‘gini’, showing the accuracy score of test data by Max_depth=3 and Max_depth=10, respectively.
# 

# In[8]:


clf_depth3 = DecisionTree(criterion='gini', max_depth=3)
clf_depth10 = DecisionTree(criterion='gini', max_depth=10)


# ### Question 2.2
# Using Max_depth=3, showing the accuracy score of test data by Criterion=‘gini’ and Criterion=’entropy’, respectively.
# 

# In[9]:


clf_gini = DecisionTree(criterion='gini', max_depth=3)
clf_entropy = DecisionTree(criterion='entropy', max_depth=3)


# - Note: All of your accuracy scores should over **0.7**
# - Note: You should get the same results when re-building the model with the same arguments,  no need to prune the trees
# - Hint: You can use the recursive method to build the nodes
# 

# ## Question 3
# Plot the [feature importance](https://sefiks.com/2020/04/06/feature-importance-in-decision-trees/) of your Decision Tree model. You can get the feature importance by counting the feature used for splitting data.
# 
# - You can simply plot the **counts of feature used** for building tree without normalize the importance. Take the figure below as example, outlook feature has been used for splitting for almost 50 times. Therefore, it has the largest importance
# 
# ![image](https://i2.wp.com/sefiks.com/wp-content/uploads/2020/04/c45-fi-results.jpg?w=481&ssl=1)

# ## Question 4
# implement the AdaBooest algorithm by using the CART you just implemented from question 2 as base learner. You should implement one arguments for the AdaBooest.
# 1. **N_estimators**: The maximum number of estimators at which boosting is terminated

# In[343]:


class AdaBoost():
    def __init__(self, n_estimators):
        return None


# In[ ]:





# ### Question 4.1
# Show the accuracy score of test data by n_estimators=10 and n_estimators=100, respectively.
# 

# In[ ]:





# ## Question 5
# implement the Random Forest algorithm by using the CART you just implemented from question 2. You should implement three arguments for the Random Forest.
# 
# 1. **N_estimators**: The number of trees in the forest. 
# 2. **Max_features**: The number of random select features to consider when looking for the best split
# 3. **Bootstrap**: Whether bootstrap samples are used when building tree
# 

# In[11]:


class RandomForest():
    def __init__(self, n_estimators, max_features, boostrap=True, criterion='gini', max_depth=None):
        return None


# ### Question 5.1
# Using Criterion=‘gini’, Max_depth=None, Max_features=sqrt(n_features), showing the accuracy score of test data by n_estimators=10 and n_estimators=100, respectively.
# 

# In[12]:


clf_10tree = RandomForest(n_estimators=10, max_features=np.sqrt(x_train.shape[1]))
clf_100tree = RandomForest(n_estimators=100, max_features=np.sqrt(x_train.shape[1]))


# In[ ]:





# ### Question 5.2
# Using Criterion=‘gini’, Max_depth=None, N_estimators=10, showing the accuracy score of test data by Max_features=sqrt(n_features) and Max_features=n_features, respectively.
# 

# In[13]:


clf_random_features = RandomForest(n_estimators=10, max_features=np.sqrt(x_train.shape[1]))
clf_all_features = RandomForest(n_estimators=10, max_features=x_train.shape[1])


# - Note: Use majority votes to get the final prediction, you may get slightly different results when re-building the random forest model

# In[ ]:





# ### Question 6.
# Try you best to get highest test accuracy score by 
# - Feature engineering
# - Hyperparameter tuning
# - Implement any other ensemble methods, such as gradient boosting. Please note that you cannot call any package. Also, only ensemble method can be used. Neural network method is not allowed to used.

# In[ ]:





# ## Supplementary
# If you have trouble to implement this homework, TA strongly recommend watching [this video](https://www.youtube.com/watch?v=LDRbO9a6XPU), which explains Decision Tree model clearly. But don't copy code from any resources, try to finish this homework by yourself! 

# In[ ]:




