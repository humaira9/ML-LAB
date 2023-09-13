#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
data=load_iris()
X=data.data
y=data.target
feature_names=data.feature_names
df=pd.DataFrame(X,columns=feature_names)
k=2
selector=SelectKBest(score_func=chi2,k=2)
X_new=selector.fit_transform(X,y)
scores=selector.scores_
p_value=selector.pvalues_
selected_indices=selector.get_support(indices=True)
for i,feature_name in enumerate(feature_names):
    print(f"Feature:{feature_name},score:{scores[i]},p-value:{p_value[i]}")


# In[15]:


selected_features=[feature_names[i] for i in selected_indices]
print(f"Selected Features:{selected_features}")


# In[18]:


X_new.shape


# In[ ]:




