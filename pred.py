import warnings
warnings.filterwarnings('ignore')
import re
import nltk
import matplotlib.pyplot as plt

import pandas as pd
import collections

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
# from sklearn.externals import joblib
import joblib
tvec = TfidfVectorizer()
lr = LogisticRegression()


# test_data.head()
comparr = ['amazon' , 'josh' , 'zs' , 'new' , 'hashed' , 'traveloka' , 'aws'  ]
# predictions = []
resarr = []

for i in range(len(comparr)):
    zero = int(0)
    one = int(0)
    test_data=pd.read_csv( comparr[i] + ".csv" , encoding='ISO-8859–1')
    model = joblib.load('model_predict.joblib')
    X_test = test_data['Review']
    # Y_test = test_data['Liked']

    predictions = model.predict(X_test)

    # from sklearn.metrics import confusion_matrix
    # confusion_matrix(predictions,test_data['Liked']) #Good model

    # zero = int(0)
    # one = int(0)
    print(comparr[i])
    # print(len(predictions))
    occurences = collections.Counter(predictions)
    zero = occurences[0]
    one = occurences[1]
    sum = one + zero
    res = (one/sum) * 100
    print(res)
    resarr.append(res)

print(resarr)
   
