---
layout: post
title:  "Predicting Occupancy of a Room, Part 1"
date:   2020-04-30 18:34:43 -0400
categories: occupancy
---
I decided to take on the [Occupancy dataset from UCI Machine Learning][occupancy]. With this dataset, I wanted to accomplish 2 tasks:

1. Predict the occupancy of a room given parameters.
2. Conduct Time Series Analysis to forecast the parameters and predict occupancy.

The tasks may sound similar, but they're actually different. The first one does not take the time into account, but rather uses the parameters or a subset of the parameters given (Light, Humidity, Temperature, CO2, etc.) to predict the occupancy of the room (binary -> 0 or 1). The second task actually does take the time into account, and forecasts the individual parameters to predict the occupancy of the room at specific times where the parameters were forecast.

In this post, I am going to show how I tackle the first task. This example uses Python 3. 
First, import the required modules and packages:
{% highlight ruby %}
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, plot_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, export_graphviz,plot_tree
from sklearn.ensemble import RandomForestClassifier
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import grangercausalitytests
from fbprophet import Prophet
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
plt.rcParams['figure.figsize'] = [10.0, 6.0]
{% endhighlight %}

Then, read in the data:
{% highlight ruby %}
# Read in data
df_train = pd.read_csv('train_data.txt')
df_val = pd.read_csv('validation_data.txt')
{% endhighlight %}

Explore the data:
```python
# Explore data
df_train.head()
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
      <th>date</th>
      <th>Temperature</th>
      <th>Humidity</th>
      <th>Light</th>
      <th>CO2</th>
      <th>HumidityRatio</th>
      <th>Occupancy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>2015-02-04 17:51:00</td>
      <td>23.18</td>
      <td>27.2720</td>
      <td>426.0</td>
      <td>721.25</td>
      <td>0.004793</td>
      <td>1</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2015-02-04 17:51:59</td>
      <td>23.15</td>
      <td>27.2675</td>
      <td>429.5</td>
      <td>714.00</td>
      <td>0.004783</td>
      <td>1</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2015-02-04 17:53:00</td>
      <td>23.15</td>
      <td>27.2450</td>
      <td>426.0</td>
      <td>713.50</td>
      <td>0.004779</td>
      <td>1</td>
    </tr>
    <tr>
      <td>4</td>
      <td>2015-02-04 17:54:00</td>
      <td>23.15</td>
      <td>27.2000</td>
      <td>426.0</td>
      <td>708.25</td>
      <td>0.004772</td>
      <td>1</td>
    </tr>
    <tr>
      <td>5</td>
      <td>2015-02-04 17:55:00</td>
      <td>23.10</td>
      <td>27.2000</td>
      <td>426.0</td>
      <td>704.50</td>
      <td>0.004757</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_train.tail()
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
      <th>date</th>
      <th>Temperature</th>
      <th>Humidity</th>
      <th>Light</th>
      <th>CO2</th>
      <th>HumidityRatio</th>
      <th>Occupancy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>8139</td>
      <td>2015-02-10 09:29:00</td>
      <td>21.05</td>
      <td>36.0975</td>
      <td>433.0</td>
      <td>787.250000</td>
      <td>0.005579</td>
      <td>1</td>
    </tr>
    <tr>
      <td>8140</td>
      <td>2015-02-10 09:29:59</td>
      <td>21.05</td>
      <td>35.9950</td>
      <td>433.0</td>
      <td>789.500000</td>
      <td>0.005563</td>
      <td>1</td>
    </tr>
    <tr>
      <td>8141</td>
      <td>2015-02-10 09:30:59</td>
      <td>21.10</td>
      <td>36.0950</td>
      <td>433.0</td>
      <td>798.500000</td>
      <td>0.005596</td>
      <td>1</td>
    </tr>
    <tr>
      <td>8142</td>
      <td>2015-02-10 09:32:00</td>
      <td>21.10</td>
      <td>36.2600</td>
      <td>433.0</td>
      <td>820.333333</td>
      <td>0.005621</td>
      <td>1</td>
    </tr>
    <tr>
      <td>8143</td>
      <td>2015-02-10 09:33:00</td>
      <td>21.10</td>
      <td>36.2000</td>
      <td>447.0</td>
      <td>821.000000</td>
      <td>0.005612</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Data seems to be ordered by date and time and appropriate for time series analysis. Time intervals seem to be about every minute. Dates are from 2015-02-04 to 2015-02-10 (6 days).


```python
df_train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 8143 entries, 1 to 8143
    Data columns (total 7 columns):
    date             8143 non-null object
    Temperature      8143 non-null float64
    Humidity         8143 non-null float64
    Light            8143 non-null float64
    CO2              8143 non-null float64
    HumidityRatio    8143 non-null float64
    Occupancy        8143 non-null int64
    dtypes: float64(5), int64(1), object(1)
    memory usage: 508.9+ KB


There are no null values in the training data. May need to convert date to datetime object. Occupancy is the only int type.


```python
df_train.describe()
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
      <th>Temperature</th>
      <th>Humidity</th>
      <th>Light</th>
      <th>CO2</th>
      <th>HumidityRatio</th>
      <th>Occupancy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>8143.000000</td>
      <td>8143.000000</td>
      <td>8143.000000</td>
      <td>8143.000000</td>
      <td>8143.000000</td>
      <td>8143.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>20.619084</td>
      <td>25.731507</td>
      <td>119.519375</td>
      <td>606.546243</td>
      <td>0.003863</td>
      <td>0.212330</td>
    </tr>
    <tr>
      <td>std</td>
      <td>1.016916</td>
      <td>5.531211</td>
      <td>194.755805</td>
      <td>314.320877</td>
      <td>0.000852</td>
      <td>0.408982</td>
    </tr>
    <tr>
      <td>min</td>
      <td>19.000000</td>
      <td>16.745000</td>
      <td>0.000000</td>
      <td>412.750000</td>
      <td>0.002674</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>19.700000</td>
      <td>20.200000</td>
      <td>0.000000</td>
      <td>439.000000</td>
      <td>0.003078</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>20.390000</td>
      <td>26.222500</td>
      <td>0.000000</td>
      <td>453.500000</td>
      <td>0.003801</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>21.390000</td>
      <td>30.533333</td>
      <td>256.375000</td>
      <td>638.833333</td>
      <td>0.004352</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>max</td>
      <td>23.180000</td>
      <td>39.117500</td>
      <td>1546.333333</td>
      <td>2028.500000</td>
      <td>0.006476</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



Occupancy seems to be binary: either 0 or 1. Assume 1 means occupied, and 0 means unoccupied.


```python
# Convert date column to datetime object
df_train['date'] = pd.to_datetime(df_train['date'])
```


```python
# Verify changed datatype for date
df_train.dtypes
```




    date             datetime64[ns]
    Temperature             float64
    Humidity                float64
    Light                   float64
    CO2                     float64
    HumidityRatio           float64
    Occupancy                 int64
    dtype: object



Date values are either at :00 or :59 seconds. Round all to nearest minute.


```python
# Set date to index and round to nearest minute for time series analysis
df_train = df_train.set_index('date')
df_train.index = df_train.index.round('min')
df_train.head()
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
      <th>Temperature</th>
      <th>Humidity</th>
      <th>Light</th>
      <th>CO2</th>
      <th>HumidityRatio</th>
      <th>Occupancy</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2015-02-04 17:51:00</td>
      <td>23.18</td>
      <td>27.2720</td>
      <td>426.0</td>
      <td>721.25</td>
      <td>0.004793</td>
      <td>1</td>
    </tr>
    <tr>
      <td>2015-02-04 17:52:00</td>
      <td>23.15</td>
      <td>27.2675</td>
      <td>429.5</td>
      <td>714.00</td>
      <td>0.004783</td>
      <td>1</td>
    </tr>
    <tr>
      <td>2015-02-04 17:53:00</td>
      <td>23.15</td>
      <td>27.2450</td>
      <td>426.0</td>
      <td>713.50</td>
      <td>0.004779</td>
      <td>1</td>
    </tr>
    <tr>
      <td>2015-02-04 17:54:00</td>
      <td>23.15</td>
      <td>27.2000</td>
      <td>426.0</td>
      <td>708.25</td>
      <td>0.004772</td>
      <td>1</td>
    </tr>
    <tr>
      <td>2015-02-04 17:55:00</td>
      <td>23.10</td>
      <td>27.2000</td>
      <td>426.0</td>
      <td>704.50</td>
      <td>0.004757</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.plotting.register_matplotlib_converters()
# Plot all columns vs. date
df_train.plot(subplots=True, figsize=(15,10))
plt.show()
```


![png](/images/output_14_0.png)


There appears to be some correlation between the variables, which can be seen from when the peaks occur. The occupancy seems less correlated to Humidity than it does with Light and CO2. Light seems to be the most important factor, since there are dips in the occupancy that coincide with dips in Light. There seems to be periodicity in the occupancy, where the occupancy can be 1 during "normal working hours" (i.e., 9am-6pm). However, there was no occupancy on 2 consecutive dates, 2015-02-07 and 2015-02-08. Perhaps those dates were the weekend, which indicates conditional seasonality. For those 2 dates, the CO2 levels were also very low with no peaks. Since the date range is small, assume no overall underlying trends in the variables. There also seems to be a high outlier in the Light data, with the value of 1546.3 within the day of 2015-02-07.

Check for Seasonality in parameters


```python
# Perform seasonality decomposition for Temperature, period = 1 day
seas_d = seasonal_decompose(df_train['Temperature'], period = (60*24))
fig=seas_d.plot()
fig.set_figheight(4)
plt.show()
```


![png](/images/output_17_0.png)



```python
# Perform seasonality decomposition for Light, period = 1 day
seas_d = seasonal_decompose(df_train['Light'], period = (60*24))
fig=seas_d.plot()
fig.set_figheight(4)
plt.show()
```


![png](/images/output_18_0.png)



```python
# Perform seasonality decomposition for CO2, period = 1 day
seas_d = seasonal_decompose(df_train['CO2'], period = (60*24))
fig=seas_d.plot()
fig.set_figheight(4)
plt.show()
```


![png](/images/output_19_0.png)



```python
# Perform seasonality decomposition for HumidityRatio, period = 1 day
seas_d = seasonal_decompose(df_train['HumidityRatio'], period = (60*24))
fig=seas_d.plot()
fig.set_figheight(4)
plt.show()
```


![png](/images/output_20_0.png)


Perform Granger Causality Tests


```python
# Examine Temperature causing Occupancy
print(grangercausalitytests(df_train[['Occupancy', 'Temperature']], maxlag=4))
```

    
    Granger Causality
    number of lags (no zero) 1
    ssr based F test:         F=10.5156 , p=0.0012  , df_denom=8139, df_num=1
    ssr based chi2 test:   chi2=10.5195 , p=0.0012  , df=1
    likelihood ratio test: chi2=10.5127 , p=0.0012  , df=1
    parameter F test:         F=10.5156 , p=0.0012  , df_denom=8139, df_num=1
    
    Granger Causality
    number of lags (no zero) 2
    ssr based F test:         F=3.5933  , p=0.0276  , df_denom=8136, df_num=2
    ssr based chi2 test:   chi2=7.1910  , p=0.0274  , df=2
    likelihood ratio test: chi2=7.1878  , p=0.0275  , df=2
    parameter F test:         F=3.5933  , p=0.0276  , df_denom=8136, df_num=2
    
    Granger Causality
    number of lags (no zero) 3
    ssr based F test:         F=1.1512  , p=0.3269  , df_denom=8133, df_num=3
    ssr based chi2 test:   chi2=3.4566  , p=0.3264  , df=3
    likelihood ratio test: chi2=3.4559  , p=0.3265  , df=3
    parameter F test:         F=1.1512  , p=0.3269  , df_denom=8133, df_num=3
    
    Granger Causality
    number of lags (no zero) 4
    ssr based F test:         F=0.6326  , p=0.6392  , df_denom=8130, df_num=4
    ssr based chi2 test:   chi2=2.5331  , p=0.6387  , df=4
    likelihood ratio test: chi2=2.5327  , p=0.6388  , df=4
    parameter F test:         F=0.6326  , p=0.6392  , df_denom=8130, df_num=4
    {1: ({'ssr_ftest': (10.515600218091105, 0.0011884805851629405, 8139.0, 1), 'ssr_chi2test': (10.51947622259464, 0.0011812295962018446, 1), 'lrtest': (10.512686480680713, 0.00118557768661934, 1), 'params_ftest': (10.515600218094972, 0.0011884805851604175, 8139.0, 1.0)}, [<statsmodels.regression.linear_model.RegressionResultsWrapper object at 0x1215440d0>, <statsmodels.regression.linear_model.RegressionResultsWrapper object at 0x12d909b90>, array([[0., 1., 0.]])]), 2: ({'ssr_ftest': (3.5932701428753764, 0.02755189101173785, 8136.0, 2), 'ssr_chi2test': (7.190956792809351, 0.027447549225782523, 2), 'lrtest': (7.1877827706048265, 0.027491143374182788, 2), 'params_ftest': (3.5932701428739904, 0.02755189101177218, 8136.0, 2.0)}, [<statsmodels.regression.linear_model.RegressionResultsWrapper object at 0x12153c350>, <statsmodels.regression.linear_model.RegressionResultsWrapper object at 0x12195b110>, array([[0., 0., 1., 0., 0.],
           [0., 0., 0., 1., 0.]])]), 3: ({'ssr_ftest': (1.1512168877249138, 0.32689209422409926, 8133.0, 3), 'ssr_chi2test': (3.456623189258871, 0.326431677123941, 3), 'lrtest': (3.455889475357253, 0.3265283318868665, 3), 'params_ftest': (1.1512168877277935, 0.32689209422298965, 8133.0, 3.0)}, [<statsmodels.regression.linear_model.RegressionResultsWrapper object at 0x121534e50>, <statsmodels.regression.linear_model.RegressionResultsWrapper object at 0x12d9090d0>, array([[0., 0., 0., 1., 0., 0., 0.],
           [0., 0., 0., 0., 1., 0., 0.],
           [0., 0., 0., 0., 0., 1., 0.]])]), 4: ({'ssr_ftest': (0.6325826140788949, 0.6392277460576581, 8130.0, 4), 'ssr_chi2test': (2.533131560141759, 0.6387130628737945, 4), 'lrtest': (2.5327374439366395, 0.6387833973927893, 4), 'params_ftest': (0.6325826140901567, 0.6392277460497593, 8130.0, 4.0)}, [<statsmodels.regression.linear_model.RegressionResultsWrapper object at 0x1216f3f10>, <statsmodels.regression.linear_model.RegressionResultsWrapper object at 0x12d8eca90>, array([[0., 0., 0., 0., 1., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 1., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 1., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 1., 0.]])])}


For a lag <= 2, the Temperature seems to cause Occupancy based on the Granger Causality Test p-value < 0.05.


```python
# Examine Humidity causing Occupancy
print(grangercausalitytests(df_train[['Occupancy', 'Humidity']], maxlag=4))
```

    
    Granger Causality
    number of lags (no zero) 1
    ssr based F test:         F=0.5782  , p=0.4471  , df_denom=8139, df_num=1
    ssr based chi2 test:   chi2=0.5784  , p=0.4469  , df=1
    likelihood ratio test: chi2=0.5784  , p=0.4470  , df=1
    parameter F test:         F=0.5782  , p=0.4471  , df_denom=8139, df_num=1
    
    Granger Causality
    number of lags (no zero) 2
    ssr based F test:         F=1.3368  , p=0.2628  , df_denom=8136, df_num=2
    ssr based chi2 test:   chi2=2.6752  , p=0.2625  , df=2
    likelihood ratio test: chi2=2.6747  , p=0.2625  , df=2
    parameter F test:         F=1.3368  , p=0.2628  , df_denom=8136, df_num=2
    
    Granger Causality
    number of lags (no zero) 3
    ssr based F test:         F=0.6158  , p=0.6047  , df_denom=8133, df_num=3
    ssr based chi2 test:   chi2=1.8490  , p=0.6043  , df=3
    likelihood ratio test: chi2=1.8488  , p=0.6044  , df=3
    parameter F test:         F=0.6158  , p=0.6047  , df_denom=8133, df_num=3
    
    Granger Causality
    number of lags (no zero) 4
    ssr based F test:         F=0.9867  , p=0.4133  , df_denom=8130, df_num=4
    ssr based chi2 test:   chi2=3.9510  , p=0.4127  , df=4
    likelihood ratio test: chi2=3.9500  , p=0.4128  , df=4
    parameter F test:         F=0.9867  , p=0.4133  , df_denom=8130, df_num=4
    {1: ({'ssr_ftest': (0.5781720070298837, 0.44705175496434246, 8139.0, 1), 'ssr_chi2test': (0.578385118716957, 0.4469459998686701, 1), 'lrtest': (0.5783645762421656, 0.4469540697217348, 1), 'params_ftest': (0.5781720070245995, 0.44705175496647276, 8139.0, 1.0)}, [<statsmodels.regression.linear_model.RegressionResultsWrapper object at 0x12d909c10>, <statsmodels.regression.linear_model.RegressionResultsWrapper object at 0x12d909b90>, array([[0., 1., 0.]])]), 2: ({'ssr_ftest': (1.336763770235409, 0.2627521241567607, 8136.0, 2), 'ssr_chi2test': (2.675170563787233, 0.26247871606533607, 2), 'lrtest': (2.6747311232975335, 0.26253639428940245, 2), 'params_ftest': (1.336763770238285, 0.2627521241561952, 8136.0, 2.0)}, [<statsmodels.regression.linear_model.RegressionResultsWrapper object at 0x12d8ecb50>, <statsmodels.regression.linear_model.RegressionResultsWrapper object at 0x12d8ecf50>, array([[0., 0., 1., 0., 0.],
           [0., 0., 0., 1., 0.]])]), 3: ({'ssr_ftest': (0.6158148460645417, 0.6046860128780425, 8133.0, 3), 'ssr_chi2test': (1.8490346171026817, 0.6043230239230211, 3), 'lrtest': (1.8488246409760904, 0.6043682147417089, 3), 'params_ftest': (0.6158148460645266, 0.6046860128780425, 8133.0, 3.0)}, [<statsmodels.regression.linear_model.RegressionResultsWrapper object at 0x12153c350>, <statsmodels.regression.linear_model.RegressionResultsWrapper object at 0x121954ad0>, array([[0., 0., 0., 1., 0., 0., 0.],
           [0., 0., 0., 0., 1., 0., 0.],
           [0., 0., 0., 0., 0., 1., 0.]])]), 4: ({'ssr_ftest': (0.9866551330898123, 0.4133429557230327, 8130.0, 4), 'ssr_chi2test': (3.950989484978097, 0.412679334524536, 4), 'lrtest': (3.9500308127244352, 0.41281068176994806, 4), 'params_ftest': (0.9866551331015199, 0.4133429557167153, 8130.0, 4.0)}, [<statsmodels.regression.linear_model.RegressionResultsWrapper object at 0x121954f50>, <statsmodels.regression.linear_model.RegressionResultsWrapper object at 0x121954c50>, array([[0., 0., 0., 0., 1., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 1., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 1., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 1., 0.]])])}


The null hypothesis fails to be rejected at the p-value of 0.05, therefore Humidity does not cause Occupancy.


```python
# Examine Light causing Occupancy
print(grangercausalitytests(df_train[['Occupancy', 'Light']], maxlag=4))
```

    
    Granger Causality
    number of lags (no zero) 1
    ssr based F test:         F=216.6264, p=0.0000  , df_denom=8139, df_num=1
    ssr based chi2 test:   chi2=216.7063, p=0.0000  , df=1
    likelihood ratio test: chi2=213.8725, p=0.0000  , df=1
    parameter F test:         F=216.6264, p=0.0000  , df_denom=8139, df_num=1
    
    Granger Causality
    number of lags (no zero) 2
    ssr based F test:         F=152.1224, p=0.0000  , df_denom=8136, df_num=2
    ssr based chi2 test:   chi2=304.4318, p=0.0000  , df=2
    likelihood ratio test: chi2=298.8777, p=0.0000  , df=2
    parameter F test:         F=152.1224, p=0.0000  , df_denom=8136, df_num=2
    
    Granger Causality
    number of lags (no zero) 3
    ssr based F test:         F=116.2704, p=0.0000  , df_denom=8133, df_num=3
    ssr based chi2 test:   chi2=349.1115, p=0.0000  , df=3
    likelihood ratio test: chi2=341.8325, p=0.0000  , df=3
    parameter F test:         F=116.2704, p=0.0000  , df_denom=8133, df_num=3
    
    Granger Causality
    number of lags (no zero) 4
    ssr based F test:         F=83.6931 , p=0.0000  , df_denom=8130, df_num=4
    ssr based chi2 test:   chi2=335.1431, p=0.0000  , df=4
    likelihood ratio test: chi2=328.4267, p=0.0000  , df=4
    parameter F test:         F=83.6931 , p=0.0000  , df_denom=8130, df_num=4
    {1: ({'ssr_ftest': (216.62642993560382, 2.0562395592856343e-48, 8139.0, 1), 'ssr_chi2test': (216.70627749547688, 4.7299069898945616e-49, 1), 'lrtest': (213.87253788191447, 1.9634670480476312e-48, 1), 'params_ftest': (216.62642993560408, 2.0562395592856343e-48, 8139.0, 1.0)}, [<statsmodels.regression.linear_model.RegressionResultsWrapper object at 0x12d909950>, <statsmodels.regression.linear_model.RegressionResultsWrapper object at 0x1215446d0>, array([[0., 1., 0.]])]), 2: ({'ssr_ftest': (152.1223915307634, 1.3784532028737795e-65, 8136.0, 2), 'ssr_chi2test': (304.4317574857288, 7.824990229010475e-67, 2), 'lrtest': (298.8777013439758, 1.2575688495486277e-65, 2), 'params_ftest': (152.12239153065397, 1.3784532030197569e-65, 8136.0, 2.0)}, [<statsmodels.regression.linear_model.RegressionResultsWrapper object at 0x121954590>, <statsmodels.regression.linear_model.RegressionResultsWrapper object at 0x121954650>, array([[0., 0., 1., 0., 0.],
           [0., 0., 0., 1., 0.]])]), 3: ({'ssr_ftest': (116.27041836773999, 1.0028907212619656e-73, 8133.0, 3), 'ssr_chi2test': (349.1114738153462, 2.323097820054367e-75, 3), 'lrtest': (341.83245601379167, 8.752597933341622e-74, 3), 'params_ftest': (116.27041836773998, 1.0028907212628492e-73, 8133.0, 3.0)}, [<statsmodels.regression.linear_model.RegressionResultsWrapper object at 0x1215eb590>, <statsmodels.regression.linear_model.RegressionResultsWrapper object at 0x121954050>, array([[0., 0., 0., 1., 0., 0., 0.],
           [0., 0., 0., 0., 1., 0., 0.],
           [0., 0., 0., 0., 0., 1., 0.]])]), 4: ({'ssr_ftest': (83.69313503029518, 9.349574743675364e-70, 8130.0, 4), 'ssr_chi2test': (335.1431370290639, 2.827331931643635e-71, 4), 'lrtest': (328.4267270025739, 7.963198645095444e-70, 4), 'params_ftest': (83.69313503028006, 9.349574743941627e-70, 8130.0, 4.0)}, [<statsmodels.regression.linear_model.RegressionResultsWrapper object at 0x12d8f2650>, <statsmodels.regression.linear_model.RegressionResultsWrapper object at 0x12d8f2290>, array([[0., 0., 0., 0., 1., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 1., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 1., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 1., 0.]])])}


Light seems to have a strong effect on Occupancy, since p-value is 0.000.


```python
# Examine CO2 causing Occupancy
print(grangercausalitytests(df_train[['Occupancy', 'CO2']], maxlag=4))
```

    
    Granger Causality
    number of lags (no zero) 1
    ssr based F test:         F=34.9532 , p=0.0000  , df_denom=8139, df_num=1
    ssr based chi2 test:   chi2=34.9661 , p=0.0000  , df=1
    likelihood ratio test: chi2=34.8912 , p=0.0000  , df=1
    parameter F test:         F=34.9532 , p=0.0000  , df_denom=8139, df_num=1
    
    Granger Causality
    number of lags (no zero) 2
    ssr based F test:         F=12.9394 , p=0.0000  , df_denom=8136, df_num=2
    ssr based chi2 test:   chi2=25.8947 , p=0.0000  , df=2
    likelihood ratio test: chi2=25.8536 , p=0.0000  , df=2
    parameter F test:         F=12.9394 , p=0.0000  , df_denom=8136, df_num=2
    
    Granger Causality
    number of lags (no zero) 3
    ssr based F test:         F=5.3197  , p=0.0012  , df_denom=8133, df_num=3
    ssr based chi2 test:   chi2=15.9727 , p=0.0011  , df=3
    likelihood ratio test: chi2=15.9571 , p=0.0012  , df=3
    parameter F test:         F=5.3197  , p=0.0012  , df_denom=8133, df_num=3
    
    Granger Causality
    number of lags (no zero) 4
    ssr based F test:         F=2.7232  , p=0.0279  , df_denom=8130, df_num=4
    ssr based chi2 test:   chi2=10.9050 , p=0.0277  , df=4
    likelihood ratio test: chi2=10.8977 , p=0.0277  , df=4
    parameter F test:         F=2.7232  , p=0.0279  , df_denom=8130, df_num=4
    {1: ({'ssr_ftest': (34.95321142853135, 3.513441792672317e-09, 8139.0, 1), 'ssr_chi2test': (34.9660950302374, 3.3549665099567356e-09, 1), 'lrtest': (34.891227760119364, 3.486481577200691e-09, 1), 'params_ftest': (34.95321142828012, 3.5134417931224022e-09, 8139.0, 1.0)}, [<statsmodels.regression.linear_model.RegressionResultsWrapper object at 0x12d909b90>, <statsmodels.regression.linear_model.RegressionResultsWrapper object at 0x121954650>, array([[0., 1., 0.]])]), 2: ({'ssr_ftest': (12.939402174169055, 2.451361363274222e-06, 8136.0, 2), 'ssr_chi2test': (25.89470823498286, 2.3825144949992743e-06, 2), 'lrtest': (25.853612705301202, 2.4319762667379205e-06, 2), 'params_ftest': (12.939402174147045, 2.4513613633286006e-06, 8136.0, 2.0)}, [<statsmodels.regression.linear_model.RegressionResultsWrapper object at 0x12d8f2b50>, <statsmodels.regression.linear_model.RegressionResultsWrapper object at 0x12d8f2410>, array([[0., 0., 1., 0., 0.],
           [0., 0., 0., 1., 0.]])]), 3: ({'ssr_ftest': (5.319664402728529, 0.0011641746096933505, 8133.0, 3), 'ssr_chi2test': (15.97272897019927, 0.0011486767474718582, 3), 'lrtest': (15.957078183288104, 0.0011571939229575265, 3), 'params_ftest': (5.3196644026789315, 0.001164174609774549, 8133.0, 3.0)}, [<statsmodels.regression.linear_model.RegressionResultsWrapper object at 0x12d911710>, <statsmodels.regression.linear_model.RegressionResultsWrapper object at 0x12d911150>, array([[0., 0., 0., 1., 0., 0., 0.],
           [0., 0., 0., 0., 1., 0., 0.],
           [0., 0., 0., 0., 0., 1., 0.]])]), 4: ({'ssr_ftest': (2.7232469552480016, 0.027863347815891896, 8130.0, 4), 'ssr_chi2test': (10.905046479096425, 0.027652144774908145, 4), 'lrtest': (10.897747430004529, 0.027737549191175867, 4), 'params_ftest': (2.723246955187612, 0.027863347818739642, 8130.0, 4.0)}, [<statsmodels.regression.linear_model.RegressionResultsWrapper object at 0x121eeddd0>, <statsmodels.regression.linear_model.RegressionResultsWrapper object at 0x12d911950>, array([[0., 0., 0., 0., 1., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 1., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 1., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 1., 0.]])])}


CO2 seems to have a strong effect on Occupancy, since p-value is < 0.05 for max_lag <= 4.


```python
# Examine HumidityRatio causing Occupancy
print(grangercausalitytests(df_train[['Occupancy', 'HumidityRatio']], maxlag=4))
```

    
    Granger Causality
    number of lags (no zero) 1
    ssr based F test:         F=2.7994  , p=0.0943  , df_denom=8139, df_num=1
    ssr based chi2 test:   chi2=2.8005  , p=0.0942  , df=1
    likelihood ratio test: chi2=2.8000  , p=0.0943  , df=1
    parameter F test:         F=2.7994  , p=0.0943  , df_denom=8139, df_num=1
    
    Granger Causality
    number of lags (no zero) 2
    ssr based F test:         F=1.7887  , p=0.1672  , df_denom=8136, df_num=2
    ssr based chi2 test:   chi2=3.5797  , p=0.1670  , df=2
    likelihood ratio test: chi2=3.5789  , p=0.1671  , df=2
    parameter F test:         F=1.7887  , p=0.1672  , df_denom=8136, df_num=2
    
    Granger Causality
    number of lags (no zero) 3
    ssr based F test:         F=0.7380  , p=0.5293  , df_denom=8133, df_num=3
    ssr based chi2 test:   chi2=2.2158  , p=0.5289  , df=3
    likelihood ratio test: chi2=2.2155  , p=0.5289  , df=3
    parameter F test:         F=0.7380  , p=0.5293  , df_denom=8133, df_num=3
    
    Granger Causality
    number of lags (no zero) 4
    ssr based F test:         F=0.9660  , p=0.4248  , df_denom=8130, df_num=4
    ssr based chi2 test:   chi2=3.8683  , p=0.4241  , df=4
    likelihood ratio test: chi2=3.8674  , p=0.4243  , df=4
    parameter F test:         F=0.9660  , p=0.4248  , df_denom=8130, df_num=4
    {1: ({'ssr_ftest': (2.7994495684347247, 0.09433510384211544, 8139.0, 1), 'ssr_chi2test': (2.8004814333696433, 0.0942360069980527, 1), 'lrtest': (2.799999924012809, 0.09426431130865497, 1), 'params_ftest': (2.799449568434711, 0.09433510384211544, 8139.0, 1.0)}, [<statsmodels.regression.linear_model.RegressionResultsWrapper object at 0x12d8ec850>, <statsmodels.regression.linear_model.RegressionResultsWrapper object at 0x12153c350>, array([[0., 1., 0.]])]), 2: ({'ssr_ftest': (1.7887487294813864, 0.16723494858875154, 8136.0, 2), 'ssr_chi2test': (3.579696019348074, 0.16698554792622738, 2), 'lrtest': (3.578909232201113, 0.167051251890557, 2), 'params_ftest': (1.7887487294809006, 0.16723494858883073, 8136.0, 2.0)}, [<statsmodels.regression.linear_model.RegressionResultsWrapper object at 0x12d911b50>, <statsmodels.regression.linear_model.RegressionResultsWrapper object at 0x12d9115d0>, array([[0., 0., 1., 0., 0.],
           [0., 0., 0., 1., 0.]])]), 3: ({'ssr_ftest': (0.7379510394387019, 0.5292573095580974, 8133.0, 3), 'ssr_chi2test': (2.2157585617967666, 0.5288510996267659, 3), 'lrtest': (2.215457044883806, 0.5289102360516149, 3), 'params_ftest': (0.7379510394391602, 0.5292573095579194, 8133.0, 3.0)}, [<statsmodels.regression.linear_model.RegressionResultsWrapper object at 0x1215348d0>, <statsmodels.regression.linear_model.RegressionResultsWrapper object at 0x12d8f2e50>, array([[0., 0., 0., 1., 0., 0., 0.],
           [0., 0., 0., 0., 1., 0., 0.],
           [0., 0., 0., 0., 0., 1., 0.]])]), 4: ({'ssr_ftest': (0.9660033188057995, 0.42478433188508447, 8130.0, 4), 'ssr_chi2test': (3.8682907806939246, 0.4241241083428684, 4), 'lrtest': (3.8673718144054874, 0.42425258322177695, 4), 'params_ftest': (0.9660033188159165, 0.4247843318793969, 8130.0, 4.0)}, [<statsmodels.regression.linear_model.RegressionResultsWrapper object at 0x12196be90>, <statsmodels.regression.linear_model.RegressionResultsWrapper object at 0x121709ad0>, array([[0., 0., 0., 0., 1., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 1., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 1., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 1., 0.]])])}


The null hypothesis fails to be rejected at the p-value of 0.05, therefore HumidityRatio does not cause Occupancy.

Check each parameter for stationarity using ADF Test


```python
# Check Temperature for stationarity
result = adfuller(df_train['Temperature'].values)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))
```

    ADF Statistic: -2.694434
    p-value: 0.075006
    Critical Values:
    	1%: -3.431
    	5%: -2.862
    	10%: -2.567


The p-value is above the rule-of-thumb threshold of 0.05, indicating that there is a unit root, and Temperature is NOT stationary.


```python
# Check Humidity for stationarity
result = adfuller(df_train['Humidity'].values)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))
```

    ADF Statistic: -0.751632
    p-value: 0.832919
    Critical Values:
    	1%: -3.431
    	5%: -2.862
    	10%: -2.567


The p-value is above the rule-of-thumb threshold of 0.05, indicating that there is a unit root, and Humidity is NOT stationary.


```python
# Check Light for stationarity
result = adfuller(df_train['Light'].values)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))
```

    ADF Statistic: -3.314650
    p-value: 0.014236
    Critical Values:
    	1%: -3.431
    	5%: -2.862
    	10%: -2.567


The p-value is below the rule-of-thumb threshold of 0.05, indicating that there is no unit root, and Light IS stationary.


```python
# Check CO2 for stationarity
result = adfuller(df_train['CO2'].values)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))
```

    ADF Statistic: -3.639003
    p-value: 0.005058
    Critical Values:
    	1%: -3.431
    	5%: -2.862
    	10%: -2.567


The p-value is below the rule-of-thumb threshold of 0.05, indicating that there is no unit root, and CO2 IS stationary.


```python
# Check HumidityRatio for stationarity
result = adfuller(df_train['HumidityRatio'].values)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))
```

    ADF Statistic: -1.279709
    p-value: 0.638403
    Critical Values:
    	1%: -3.431
    	5%: -2.862
    	10%: -2.567


The p-value is above the rule-of-thumb threshold of 0.05, indicating that there is a unit root, and HumidityRatio is NOT stationary.


```python
# Examine hours for Occupancy=1
df_train[df_train.index.day == 5]['Occupancy'].plot()
plt.show()
```


![png](/images/output_43_0.png)



```python
df_train[df_train.index.day == 6]['Occupancy'].plot()
plt.show()
```


![png](/images/output_44_0.png)



```python
df_train[df_train.index.day == 9]['Occupancy'].plot()
plt.show()
```


![png](/images/output_45_0.png)


The plots confirm that the hours for occupancy tend to be roughly between 9am-6pm.

Since the day of the week and time of the day seem to be important factors in predicting occupancy and trends in the other variables, create new columns for "WorkHours" and "Weekday"


```python
# Separate X and y
X_df_train = df_train.iloc[:,:5]
y_df_train = df_train['Occupancy']
```


```python
# Add binary column for Weekday
X_df_train['Weekday'] = np.where((X_df_train.index.day == 7) | (X_df_train.index.day == 8), 0, 1)
```


```python
# Add binary column for WorkHours
X_df_train['WorkHours'] = np.where((X_df_train.index.hour >= 8) & (X_df_train.index.day <= 6), 1, 0)
```


```python
X_df_train
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
      <th>Temperature</th>
      <th>Humidity</th>
      <th>Light</th>
      <th>CO2</th>
      <th>HumidityRatio</th>
      <th>Weekday</th>
      <th>WorkHours</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2015-02-04 17:51:00</td>
      <td>23.18</td>
      <td>27.2720</td>
      <td>426.0</td>
      <td>721.250000</td>
      <td>0.004793</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>2015-02-04 17:52:00</td>
      <td>23.15</td>
      <td>27.2675</td>
      <td>429.5</td>
      <td>714.000000</td>
      <td>0.004783</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>2015-02-04 17:53:00</td>
      <td>23.15</td>
      <td>27.2450</td>
      <td>426.0</td>
      <td>713.500000</td>
      <td>0.004779</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>2015-02-04 17:54:00</td>
      <td>23.15</td>
      <td>27.2000</td>
      <td>426.0</td>
      <td>708.250000</td>
      <td>0.004772</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>2015-02-04 17:55:00</td>
      <td>23.10</td>
      <td>27.2000</td>
      <td>426.0</td>
      <td>704.500000</td>
      <td>0.004757</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>2015-02-10 09:29:00</td>
      <td>21.05</td>
      <td>36.0975</td>
      <td>433.0</td>
      <td>787.250000</td>
      <td>0.005579</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2015-02-10 09:30:00</td>
      <td>21.05</td>
      <td>35.9950</td>
      <td>433.0</td>
      <td>789.500000</td>
      <td>0.005563</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2015-02-10 09:31:00</td>
      <td>21.10</td>
      <td>36.0950</td>
      <td>433.0</td>
      <td>798.500000</td>
      <td>0.005596</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2015-02-10 09:32:00</td>
      <td>21.10</td>
      <td>36.2600</td>
      <td>433.0</td>
      <td>820.333333</td>
      <td>0.005621</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2015-02-10 09:33:00</td>
      <td>21.10</td>
      <td>36.2000</td>
      <td>447.0</td>
      <td>821.000000</td>
      <td>0.005612</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>8143 rows × 7 columns</p>
</div>




```python
# Examine outlier for Light
X_df_train[X_df_train.index.day == 7]['Light'].plot(grid=True)
plt.show()
```


![png](/images/output_52_0.png)


This is most likely an error, so we'll assume that the data for that spike can be removed and imputed.


```python
# Find time when spike occurred
X_df_train[(X_df_train.index.day == 7) & (X_df_train.index.hour == 9) & (X_df_train['Light'] > 400)]['Light']
```




    date
    2015-02-07 09:41:00     611.500000
    2015-02-07 09:42:00    1546.333333
    2015-02-07 09:43:00    1451.750000
    2015-02-07 09:44:00     829.000000
    Name: Light, dtype: float64



The data for Light was unusually high for 4 data points (09:40:59 through 09:43:59). To improve training the model, we'll change these values by imputing using simple linear regression.


```python
# Replace high values with NaN
X_df_train.at['2015-02-07 09:41:00', 'Light'] = np.nan
X_df_train.at['2015-02-07 09:42:00', 'Light'] = np.nan
X_df_train.at['2015-02-07 09:43:00', 'Light'] = np.nan
X_df_train.at['2015-02-07 09:44:00', 'Light'] = np.nan
```


```python
# Interpolate NaN values using simple linear regression
X_df_train['Light'].interpolate(method='linear', inplace=True)
```


```python
# Verify values were imputed
X_df_train[X_df_train.index.day == 7]['Light'].plot(grid=True)
plt.show()
```


![png](/images/output_58_0.png)


CO2 also had some unusually high spikes near the end of the day on 2015-02-09. Let's zoom in and examine it.


```python
# Examine spikes for CO2
df_train[df_train.index.day == 9]['CO2'].plot(grid=True)
plt.show()
```


![png](/images/output_60_0.png)


However, the spike for CO2 is still in the range of the data, so we'll keep it and not bother smoothing it for now.


```python
# Examine correlation matrix for predicting variables
X_df_train.corr()
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
      <th>Temperature</th>
      <th>Humidity</th>
      <th>Light</th>
      <th>CO2</th>
      <th>HumidityRatio</th>
      <th>Weekday</th>
      <th>WorkHours</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Temperature</td>
      <td>1.000000</td>
      <td>-0.141759</td>
      <td>0.654502</td>
      <td>0.559894</td>
      <td>0.151762</td>
      <td>0.418657</td>
      <td>0.532213</td>
    </tr>
    <tr>
      <td>Humidity</td>
      <td>-0.141759</td>
      <td>1.000000</td>
      <td>0.040864</td>
      <td>0.439023</td>
      <td>0.955198</td>
      <td>0.108551</td>
      <td>-0.327712</td>
    </tr>
    <tr>
      <td>Light</td>
      <td>0.654502</td>
      <td>0.040864</td>
      <td>1.000000</td>
      <td>0.670003</td>
      <td>0.234770</td>
      <td>0.284592</td>
      <td>0.404677</td>
    </tr>
    <tr>
      <td>CO2</td>
      <td>0.559894</td>
      <td>0.439023</td>
      <td>0.670003</td>
      <td>1.000000</td>
      <td>0.626556</td>
      <td>0.394834</td>
      <td>0.201954</td>
    </tr>
    <tr>
      <td>HumidityRatio</td>
      <td>0.151762</td>
      <td>0.955198</td>
      <td>0.234770</td>
      <td>0.626556</td>
      <td>1.000000</td>
      <td>0.243146</td>
      <td>-0.167921</td>
    </tr>
    <tr>
      <td>Weekday</td>
      <td>0.418657</td>
      <td>0.108551</td>
      <td>0.284592</td>
      <td>0.394834</td>
      <td>0.243146</td>
      <td>1.000000</td>
      <td>0.462569</td>
    </tr>
    <tr>
      <td>WorkHours</td>
      <td>0.532213</td>
      <td>-0.327712</td>
      <td>0.404677</td>
      <td>0.201954</td>
      <td>-0.167921</td>
      <td>0.462569</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



The correlation matrix shows that Humidity and HumidityRatio are highly correlated, so one of these may potentially be removed from the model to reduce multicollinearity. CO2 and HumidityRatio seem somewhat correlated, too. Therefore, let us drop HumidityRatio from the model.


```python
X_df_train = X_df_train.drop(columns = 'HumidityRatio')
```


```python
# Examine correlation matrix after dropping HumidityRatio
X_df_train.corr()
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
      <th>Temperature</th>
      <th>Humidity</th>
      <th>Light</th>
      <th>CO2</th>
      <th>Weekday</th>
      <th>WorkHours</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Temperature</td>
      <td>1.000000</td>
      <td>-0.141759</td>
      <td>0.654502</td>
      <td>0.559894</td>
      <td>0.418657</td>
      <td>0.532213</td>
    </tr>
    <tr>
      <td>Humidity</td>
      <td>-0.141759</td>
      <td>1.000000</td>
      <td>0.040864</td>
      <td>0.439023</td>
      <td>0.108551</td>
      <td>-0.327712</td>
    </tr>
    <tr>
      <td>Light</td>
      <td>0.654502</td>
      <td>0.040864</td>
      <td>1.000000</td>
      <td>0.670003</td>
      <td>0.284592</td>
      <td>0.404677</td>
    </tr>
    <tr>
      <td>CO2</td>
      <td>0.559894</td>
      <td>0.439023</td>
      <td>0.670003</td>
      <td>1.000000</td>
      <td>0.394834</td>
      <td>0.201954</td>
    </tr>
    <tr>
      <td>Weekday</td>
      <td>0.418657</td>
      <td>0.108551</td>
      <td>0.284592</td>
      <td>0.394834</td>
      <td>1.000000</td>
      <td>0.462569</td>
    </tr>
    <tr>
      <td>WorkHours</td>
      <td>0.532213</td>
      <td>-0.327712</td>
      <td>0.404677</td>
      <td>0.201954</td>
      <td>0.462569</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



Inspect df_val:


```python
df_val.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 9752 entries, 1 to 9752
    Data columns (total 7 columns):
    date             9752 non-null object
    Temperature      9752 non-null float64
    Humidity         9752 non-null float64
    Light            9752 non-null float64
    CO2              9752 non-null float64
    HumidityRatio    9752 non-null float64
    Occupancy        9752 non-null int64
    dtypes: float64(5), int64(1), object(1)
    memory usage: 609.5+ KB


Again, no null values.


```python
df_val.describe()
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
      <th>Temperature</th>
      <th>Humidity</th>
      <th>Light</th>
      <th>CO2</th>
      <th>HumidityRatio</th>
      <th>Occupancy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>9752.000000</td>
      <td>9752.000000</td>
      <td>9752.000000</td>
      <td>9752.000000</td>
      <td>9752.000000</td>
      <td>9752.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>21.001768</td>
      <td>29.891910</td>
      <td>123.067930</td>
      <td>753.224832</td>
      <td>0.004589</td>
      <td>0.210111</td>
    </tr>
    <tr>
      <td>std</td>
      <td>1.020693</td>
      <td>3.952844</td>
      <td>208.221275</td>
      <td>297.096114</td>
      <td>0.000531</td>
      <td>0.407408</td>
    </tr>
    <tr>
      <td>min</td>
      <td>19.500000</td>
      <td>21.865000</td>
      <td>0.000000</td>
      <td>484.666667</td>
      <td>0.003275</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>20.290000</td>
      <td>26.642083</td>
      <td>0.000000</td>
      <td>542.312500</td>
      <td>0.004196</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>20.790000</td>
      <td>30.200000</td>
      <td>0.000000</td>
      <td>639.000000</td>
      <td>0.004593</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>21.533333</td>
      <td>32.700000</td>
      <td>208.250000</td>
      <td>831.125000</td>
      <td>0.004998</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>max</td>
      <td>24.390000</td>
      <td>39.500000</td>
      <td>1581.000000</td>
      <td>2076.500000</td>
      <td>0.005769</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Convert date column to datetime object and set as index, round to nearest minute
df_val['date'] =  pd.to_datetime(df_val['date'])
df_val = df_val.set_index('date')
df_val.index = df_val.index.round('min')

# Plot df_val to visualize
df_val.plot(subplots=True, figsize=(15,10))
plt.show()
```


![png](/images/output_70_0.png)


It can be seen that this is a continuation of the time series data from the training set. Now, the dates 2015-02-11 to 2015-02-19 are given. The same seasonality pattern of no occupancy for 2 consecutive days (2015-02-14 and 2015-02-15) can be observed, which indicates conditional seasonality. Add columns for Weekday or Weekend and Hour to take into account this seasonality for predicting Occupancy. Also, note that the time range is quite short here, and none of the variables lead us to predict that there is a trend component to this time series. Therefore, assume there is no trend component.


```python
# Drop HumidityRatio from df_val
df_val = df_val.drop(columns = 'HumidityRatio')
```


```python
# Separate X and y
X_df_val = df_val.iloc[:,:4]
y_df_val = df_val['Occupancy']
```


```python
# Add columns for Weekday and WorkHours
X_df_val['Weekday'] = np.where((X_df_val.index.day == 14) | (X_df_val.index.day == 15), 0, 1)
X_df_val['WorkHours'] = np.where((X_df_val.index.hour >= 8) & (X_df_val.index.hour <= 6), 1, 0)
```

Split the validation set into validation set for comparing models and test set for assessing model accuracy. Use dates from 2015-02-11 to 2015-02-14 for validation set, and 2015-02-15 to 2015-02-18 for test set. This gives a good balance by including a date in each set with a "weekend" day, and makes the test set the latest set in time.


```python
X_df_test = X_df_val[df_val.index.day >= 15]
y_df_test = y_df_val[df_val.index.day >= 15]
X_df_val = X_df_val[df_val.index.day <= 14]
y_df_val = y_df_val[df_val.index.day <= 14]
```


```python
X_df_val.head()
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
      <th>Temperature</th>
      <th>Humidity</th>
      <th>Light</th>
      <th>CO2</th>
      <th>Weekday</th>
      <th>WorkHours</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2015-02-11 14:48:00</td>
      <td>21.7600</td>
      <td>31.133333</td>
      <td>437.333333</td>
      <td>1029.666667</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2015-02-11 14:49:00</td>
      <td>21.7900</td>
      <td>31.000000</td>
      <td>437.333333</td>
      <td>1000.000000</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2015-02-11 14:50:00</td>
      <td>21.7675</td>
      <td>31.122500</td>
      <td>434.000000</td>
      <td>1003.750000</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2015-02-11 14:51:00</td>
      <td>21.7675</td>
      <td>31.122500</td>
      <td>439.000000</td>
      <td>1009.500000</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2015-02-11 14:52:00</td>
      <td>21.7900</td>
      <td>31.133333</td>
      <td>437.333333</td>
      <td>1005.666667</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
y_df_val.head()
```




    date
    2015-02-11 14:48:00    1
    2015-02-11 14:49:00    1
    2015-02-11 14:50:00    1
    2015-02-11 14:51:00    1
    2015-02-11 14:52:00    1
    Name: Occupancy, dtype: int64




```python
X_df_test.head()
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
      <th>Temperature</th>
      <th>Humidity</th>
      <th>Light</th>
      <th>CO2</th>
      <th>Weekday</th>
      <th>WorkHours</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2015-02-15 00:00:00</td>
      <td>19.89</td>
      <td>35.745</td>
      <td>0.0</td>
      <td>535.500000</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2015-02-15 00:01:00</td>
      <td>19.89</td>
      <td>35.700</td>
      <td>0.0</td>
      <td>541.000000</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2015-02-15 00:02:00</td>
      <td>20.00</td>
      <td>35.700</td>
      <td>0.0</td>
      <td>539.333333</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2015-02-15 00:03:00</td>
      <td>20.00</td>
      <td>35.700</td>
      <td>0.0</td>
      <td>541.000000</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2015-02-15 00:04:00</td>
      <td>20.00</td>
      <td>35.700</td>
      <td>0.0</td>
      <td>542.000000</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
y_df_test.head()
```




    date
    2015-02-15 00:00:00    0
    2015-02-15 00:01:00    0
    2015-02-15 00:02:00    0
    2015-02-15 00:03:00    0
    2015-02-15 00:04:00    0
    Name: Occupancy, dtype: int64




```python
# Normalize appropriate columns for building Classification Models
cols_to_norm = ['Temperature', 'Humidity', 'Light', 'CO2']
X_df_train[cols_to_norm] = X_df_train[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
```


```python
X_df_val[cols_to_norm] = X_df_val[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
X_df_test[cols_to_norm] = X_df_test[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
```


```python
X_train = X_df_train
y_train = y_df_train
X_test = X_df_val
y_test = y_df_val
```


```python
# Implement Logistic Regression
logreg = LogisticRegression(solver='lbfgs', max_iter=200)
# Fit model: Predict y from x_test after training on x_train and y_train
y_pred = logreg.fit(X_train, y_train).predict(X_test)
# Report testing accuracy
print("Testing accuracy out of a total %d points : %f" % (X_test.shape[0], accuracy_score(y_test, y_pred)))
```

    Testing accuracy out of a total 4872 points : 0.811576



```python
# Implement the Gaussian Naive Bayes algorithm for classification
gnb = GaussianNB()
# Fit model: Predict y from x_test after training on x_train and y_train
y_pred = gnb.fit(X_train, y_train).predict(X_test)
# Report testing accuracy
print("Testing accuracy out of a total %d points : %f" % (X_test.shape[0], accuracy_score(y_test, y_pred)))
```

    Testing accuracy out of a total 4872 points : 0.888547



```python
# Implement KNN
neigh = KNeighborsClassifier(n_neighbors=3) #Best: n_neighbors=3
# Fit model: Predict y from x_test after training on x_train and y_train
y_pred = neigh.fit(X_train, y_train).predict(X_test)
# Report testing accuracy
print("Testing accuracy out of a total %d points : %f" % (X_test.shape[0], accuracy_score(y_test, y_pred)))
```

    Testing accuracy out of a total 4872 points : 0.932471



```python
# Build classifier using svm
SVM = svm.SVC(C=1, kernel = 'rbf').fit(X_train, y_train) #Best: C=1, kernel = 'rbf'
y_pred = SVM.predict(X_test)
print("Testing accuracy out of a total %d points : %f" % (X_test.shape[0], accuracy_score(y_test, y_pred)))
```

    Testing accuracy out of a total 4872 points : 0.991379



```python
# Build classifier using simple neural network
NN = MLPClassifier(solver = 'adam', learning_rate_init = 0.001, max_iter = 250, hidden_layer_sizes=(5, 2), random_state=99).fit(X_train, y_train)
y_pred = NN.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Testing accuracy out of a total %d points : %f" % (X_test.shape[0], accuracy_score(y_test, y_pred)))
```

    Testing accuracy out of a total 4872 points : 0.837233



```python
# Build classifier using CART
dct = DecisionTreeClassifier(max_depth=5, random_state=99)
dct.fit(X_train, y_train)
y_pred = dct.predict(X_test)
print("Testing accuracy out of a total %d points : %f" % (X_test.shape[0], accuracy_score(y_test, y_pred)))
```

    Testing accuracy out of a total 4872 points : 0.880747



```python
# Plot decision tree to view feature importances
plt.figure(figsize=(25,10)) #Set figure size for legibility
plot_tree(dct, filled=True)
plt.show()
```


![png](/images/output_90_0.png)


The Decision Tree plot shows that X[2] (Light) is the most important feature, followed by X[3] (CO2) and X[0] (Temperature).


```python
# Build classifier using Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=99)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print("Testing accuracy out of a total %d points : %f" % (X_test.shape[0], accuracy_score(y_test, y_pred)))
```

    Testing accuracy out of a total 4872 points : 0.838259



```python
# Calculate feature importances
importances = rf.feature_importances_
print(importances)

# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]
print(indices)
```

    [0.06784078 0.02669683 0.64817013 0.21166027 0.01187343 0.03375856]
    [2 3 0 5 1 4]


The rank and weights of the features from the Random Forest classifier confirm that X[2] (Light) is the most important feature.

#### SVM model with C=1 and kernel='rbf' performed the best on the validation set. Assess the true accuracy of the model using the test set.


```python
X_test = X_df_test
y_test = y_df_test
```


```python
# Build classifier using svm
SVM = svm.SVC(C=1, kernel = 'rbf').fit(X_train, y_train) #Best: C=1, kernel = 'rbf'
y_pred = SVM.predict(X_test)
print("Testing accuracy out of a total %d points : %f" % (X_test.shape[0], accuracy_score(y_test, y_pred)))
```

    Testing accuracy out of a total 4880 points : 0.996107


The accuracy on the test set is **0.996107**, which is very high!


```python
# Plot confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(SVM, X_test, y_test,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)
```

    Confusion matrix, without normalization
    [[3765   18]
     [   1 1096]]
    Normalized confusion matrix
    [[9.95241872e-01 4.75812847e-03]
     [9.11577028e-04 9.99088423e-01]]



![png](/images/output_99_1.png)



![png](/images/output_99_2.png)


#### Conclusion
The model only predicts based on the variables Temperature, Humidity, CO2, Weekday, and WorkHours. It assumes that the information on the variables is known. For further exploration, subsets of the variables could be used to train more models. For a more difficult problem, forecast the variables and make predictions from the forecast, and compare the predictions with the actual results.




[occupancy]: https://archive.ics.uci.edu/ml/datasets/Occupancy+Detection+