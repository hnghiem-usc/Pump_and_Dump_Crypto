#!/usr/bin/env python
# coding: utf-8

# In this module, we implemnent a Grid Search using materials to make sure we have the materials to determine the best parameters to train the Neural Network model for Binary Classfication.
# 
# This model is considered the __BASELINE__ for prediction.

# In[17]:


import tensorflow as tf


# In[18]:


from keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector, Bidirectional, LeakyReLU, ReLU
from keras.models import Model, Sequential
from keras import regularizers
from keras.backend import clear_session as keras_clear_session
from keras.optimizers import Adam
import keras.backend as KB


# In[19]:


import os, sys
print(os.getcwd())
sys.path.append("../Before_Announcement_Analysis/LSTM Model")


# In[20]:


from time_series_utills import *


# In[21]:


sys.path.append("../Before_Announcement_Analysis")
from analysis_utils import * 
from numpy import newaxis
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.metrics import f1_score, make_scorer

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')


# In[22]:


from keras.wrappers.scikit_learn import KerasClassifier


# # I. Import Data

# In[23]:


data = pd.read_pickle("ohlcv_data_slide_scaled_h.pkl")
data.shape


# In[24]:


#Import the column names
ohlcv_dict = pd.read_pickle("ohlcv_dict_h.pkl")


# In[25]:


ohlcv_features = sum([v for k,v in ohlcv_dict.items() if k in ['open','high','low','close','volumefrom','volumeto']],[])


# In[26]:


#Selecting only OHLCV data for trainin
ohlcv_features.remove('open_h0')
ohlcv_features.remove('high_h0')
ohlcv_features.remove('low_h0')
ohlcv_features.remove('close_h0')
ohlcv_features.remove('volumefrom_h0')
ohlcv_features.remove('volumeto_h0')


# In[27]:


d = data[ohlcv_features + ['pumped_yn']]


# # II. Reshape data

# In[28]:


## Split data first
x_train,x_test,y_train,y_test = train_test_split(d[ohlcv_features],data['pumped_yn'],test_size = 0.3,shuffle=False)
print("x_train has shape:",x_train.shape
     ,"\ny_train has shape:",y_train.shape
     ,"\nx_test has shape:",x_test.shape
     ,"\ny_test has shape:",y_test.shape)


# In[29]:


# Force the date cut off according to index. Make sure it's consistency with the other trainign data.


# In[30]:


#Create the group dictionary to be processed for the functions.
grp_dict = dict()
for root in ["open","high","low","close","volumefrom","volumeto"]:
    grp_dict[root] = data[ohlcv_features].filter(regex=root).columns.tolist()


# In[31]:


# x_train, col_dict = reshape_to_timeseries(x_train,grp_dict)
# x_test, _ = reshape_to_timeseries(x_test,grp_dict)


# In[32]:


dim_1 = x_train.shape[1]
# dim_2 = x_train.shape[2]
print(dim_1)


# # III. Classfication Model 

# In[33]:


##Define loss function for F1 as metrics
def get_f1(y_true, y_pred): #taken from old keras source code
    tp = KB.sum(KB.round(KB.clip(y_true*y_pred,0,1)))
    #Possible positivse
    pp = KB.sum(KB.round(KB.clip(y_true,0,1)))
    #predicted positive
    predp = KB.sum(KB.round(KB.clip(y_pred,0,1)))
    #Calculate Precision and Recall
    precision = tp/(predp+KB.epsilon())
    recall = tp/(pp + KB.epsilon())
    f1_score = 2*(precision*recall)/(precision+recall+KB.epsilon())
    return f1_score


# In[34]:


def create_OHLCV_model(learn_rate=0.001, layers=[200,200],metrics_fn=get_f1):
    '''
    Builds a model
    '''
    model = Sequential()
    model.reset_states()
    for index, layer in enumerate(layers): 
        if not index:
            model.add((Dense(layer,input_dim=dim_1,activation="relu")))
        else:
            model.add((Dense(layer,activation="relu")))
    model.add((Dense(1,activation="sigmoid")))
    #optimize
    opt = Adam(learning_rate=learn_rate)
    model.compile(loss="binary_crossentropy",optimizer=opt)
    model.summary()
    return model


# In[35]:


model = KerasClassifier(create_OHLCV_model, verbose=True)


# In[36]:


#Define grid search parameters
layers =[[100,100],[100,200],[100,400],[200,100],[200,200],[200,400]]
learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
batch_size = [10, 50, 100,500,1000]
epochs = [10, 20, 50]


# In[37]:


params_dict = dict(layers=layers,learn_rate=learn_rate,batch_size=batch_size,epochs=epochs)


# ## 2a. Train

# In[38]:


KB.clear_session()
model = KerasClassifier(build_fn=create_OHLCV_model, verbose=0)


# In[39]:


#Set seed for reproducibility
seed = 123
np.random.seed(seed)


# In[40]:


f1_scorer = make_scorer(f1_score)


# In[41]:


grid_cv = GridSearchCV(estimator=model,param_grid=params_dict,scoring=f1_scorer,n_jobs=-1,cv=3)


# In[42]:


grid_cv.fit(x_train,y_train)


# ## 2b. Best Parameters

# In[ ]:


grid_cv.best_params_


# In[ ]:


grid_cv.cv_results_.keys()


# In[ ]:


print("Mean Test Score", grid_cv.cv_results_["mean_test_score"])


# In[ ]:


print("Best: %f using %s" % (grid_cv.best_score_, grid_cv.best_params_))


# In[ ]:


for index,layer in enumerate([100,200,400]):
    if not index: 
        print()
    else: 
        print("else", index)


# In[ ]:




