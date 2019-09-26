
# Heart Disease Decision Tree Classifier Model

#### This database contains 76 attributes, but all published experiments refer to using a subset of 14 of them. In particular, the Cleveland database is the only one that has been used by ML researchers to this date. The "target" field refers to the presence of heart disease in the patient.


```python
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
%matplotlib inline
```


```python
# the heart.csv file can be found in the github along with this notebook
df = pd.read_csv('heart.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>sex</th>
      <th>cp</th>
      <th>trestbps</th>
      <th>chol</th>
      <th>fbs</th>
      <th>restecg</th>
      <th>thalach</th>
      <th>exang</th>
      <th>oldpeak</th>
      <th>slope</th>
      <th>ca</th>
      <th>thal</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>63</td>
      <td>1</td>
      <td>3</td>
      <td>145</td>
      <td>233</td>
      <td>1</td>
      <td>0</td>
      <td>150</td>
      <td>0</td>
      <td>2.3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>37</td>
      <td>1</td>
      <td>2</td>
      <td>130</td>
      <td>250</td>
      <td>0</td>
      <td>1</td>
      <td>187</td>
      <td>0</td>
      <td>3.5</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>41</td>
      <td>0</td>
      <td>1</td>
      <td>130</td>
      <td>204</td>
      <td>0</td>
      <td>0</td>
      <td>172</td>
      <td>0</td>
      <td>1.4</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>56</td>
      <td>1</td>
      <td>1</td>
      <td>120</td>
      <td>236</td>
      <td>0</td>
      <td>1</td>
      <td>178</td>
      <td>0</td>
      <td>0.8</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>57</td>
      <td>0</td>
      <td>0</td>
      <td>120</td>
      <td>354</td>
      <td>0</td>
      <td>1</td>
      <td>163</td>
      <td>1</td>
      <td>0.6</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['target'].value_counts()
```




    1    165
    0    138
    Name: target, dtype: int64




```python
df['target'] = df['target'].replace(0, 'no');
```


```python
df['target'] = df['target'].replace(1, 'yes');
```


```python
df = df.rename(columns={'target': 'heart_disease'})
```


```python
# making sure that our values are in the correct form for our shift to a numpy array
df.dtypes
```




    age                int64
    sex                int64
    cp                 int64
    trestbps           int64
    chol               int64
    fbs                int64
    restecg            int64
    thalach            int64
    exang              int64
    oldpeak          float64
    slope              int64
    ca                 int64
    thal               int64
    heart_disease     object
    dtype: object



### We now set up the panda df as a numpy array before processing. X will represent the independent vars while y will solely have the target column.


```python
#note the one-hot encoding was already present in the data set so didn't need to be done
X = df[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']].values
X[0:5]
```




    array([[ 63. ,   1. ,   3. , 145. , 233. ,   1. ,   0. , 150. ,   0. ,
              2.3,   0. ,   0. ,   1. ],
           [ 37. ,   1. ,   2. , 130. , 250. ,   0. ,   1. , 187. ,   0. ,
              3.5,   0. ,   0. ,   2. ],
           [ 41. ,   0. ,   1. , 130. , 204. ,   0. ,   0. , 172. ,   0. ,
              1.4,   2. ,   0. ,   2. ],
           [ 56. ,   1. ,   1. , 120. , 236. ,   0. ,   1. , 178. ,   0. ,
              0.8,   2. ,   0. ,   2. ],
           [ 57. ,   0. ,   0. , 120. , 354. ,   0. ,   1. , 163. ,   1. ,
              0.6,   2. ,   0. ,   2. ]])




```python
y = df[['heart_disease']]
y[0:5]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>heart_disease</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>yes</td>
    </tr>
    <tr>
      <th>1</th>
      <td>yes</td>
    </tr>
    <tr>
      <th>2</th>
      <td>yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>yes</td>
    </tr>
    <tr>
      <th>4</th>
      <td>yes</td>
    </tr>
  </tbody>
</table>
</div>



### It's now time for the modelling. We have to start off by preparing our test/train splits to work on our decision tree.


```python
from sklearn.model_selection import train_test_split
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 3)
#check size of each set
X_train.shape
y_train.shape
X_test.shape
y_test.shape
```




    (91, 1)



##### The model will be a decision tree classifier where we will be using entropy to figure out the order of the columns to be used in making decisions. We will also fit the model to the train sets in order for the model to learn how to make its decisions


```python
heartTree = DecisionTreeClassifier(criterion="entropy")
heartTree
```




    DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
                           max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, presort=False,
                           random_state=None, splitter='best')




```python
heartTree.fit(X_train,y_train)
```




    DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
                           max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, presort=False,
                           random_state=None, splitter='best')



#### It's time for predictions! Using the heartTree model we will get an numpy array of the yhat predictions based on the X_test set


```python
predTree = heartTree.predict(X_test)
```


```python
print (predTree[0:5])
print (y_test[0:5])
```

    ['yes' 'yes' 'yes' 'yes' 'yes']
        heart_disease
    245            no
    162           yes
    10            yes
    161           yes
    73            yes


#### It seems like the model is not 100% accurate but lets check the accuracy score to get an idea of all the predictions


```python
from sklearn import metrics
import matplotlib.pyplot as plt
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test, predTree))
```

    DecisionTrees's Accuracy:  0.8021978021978022

