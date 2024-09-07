##### Thursday the 5th of September, 2024
# BCHydro Energy Consumption Forecast with XGBoost

## Introduction
### Background

In British Columbia, BCHydro supplies over 95% of the population's electricity, primarily through renewable hydroelectric power. As the province's main energy provider, the utility company plays a key role in maintaining reliable energy access, powering industries, and sustaining vital infrastructure, all while promoting environmental stewardship and energy efficiency.

The ability for BCHydro to execute reliable and accurate energy demand forecasting is essential for efficient management and planning of power generation, distribution, and consumption. In general, predicting energy demand is a challenging task, as it is shaped by various factors such as weather conditions, economic trends, and societal behaviors. This project explores the development of a forecasting model using machine learning techniques to predict future energy demand.

### Objective

 The goal of this project is to employ the XGBoost library and cross-validation techniques in developing a machine learning model that can predict/estimate future energy consumption levels based on historical BCHydro time-series data. 


## Dataset description
The dataset used in this project is called [The Hourly Usage of Energy Dataset for Buildings in British Columbia](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/N3HGRN&version=5.0&selectTab=termsTab).

This dataset consists of hourly energy consumption data in kilowatts per hour, donated by residential customers of BCHydro, the primary provincial power utility company of British Columbia, Canada. The sample contains data from 22 homes, the majority of which have 3 years of historical consumption data as limited by the BCHydro customer portal from which the data was sourced.

The date range in which data is present is from June 1, 2012 to May 19, 2020

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>energy_kWh</th>
    </tr>
    <tr>
      <th>datetime</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2012-06-01 01:00:00</th>
      <td>1.011</td>
    </tr>
    <tr>
      <th>2012-06-01 02:00:00</th>
      <td>0.451</td>
    </tr>
    <tr>
      <th>2012-06-01 03:00:00</th>
      <td>0.505</td>
    </tr>
    <tr>
      <th>2012-06-01 04:00:00</th>
      <td>0.441</td>
    </tr>
    <tr>
      <th>2012-06-01 05:00:00</th>
      <td>0.468</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>2020-05-19 19:00:00</th>
      <td>3.060</td>
    </tr>
    <tr>
      <th>2020-05-19 20:00:00</th>
      <td>2.940</td>
    </tr>
    <tr>
      <th>2020-05-19 21:00:00</th>
      <td>1.970</td>
    </tr>
    <tr>
      <th>2020-05-19 22:00:00</th>
      <td>2.130</td>
    </tr>
    <tr>
      <th>2020-05-19 23:00:00</th>
      <td>1.010</td>
    </tr>
  </tbody>
</table>
</div>


----


## Python Modules
The following python libraries were used in this project:

- pandas
- numpy
- matplotlib
- seaborn
- Scikit-learn
- datetime
- math

---
## Exploratory Data Analysis
### Data Visualization
First, let's take a look at all of the energy consumption data across all available dates:

    
![png](visualizations/bchydro-forecast_11_0.png)
    


---
## Time Series Cross-Validation
    
![png](visualizations/bchydro-forecast_14_0.png)
    


---
## Feature Engineering
For a more robust predictive analysis, we create and employ a function to add a number of time-based features to our data, as derived from the datetime index in our dataframe. The added features are as follows:
- `hour`
- `dayofweek`
- `quarter`
- `month`
- `year`
- `dayofyear`

The code below creates our function:


```python
def make_features(data):
    data = data.copy()
    data["hour"] = data.index.hour
    data["dayofweek"] = data.index.dayofweek
    data["quarter"] = data.index.quarter
    data["month"] = data.index.month
    data["year"] = data.index.year
    data["dayofyear"] = data.index.dayofyear
    return data
```


### Visualizing Feature Relationships

Next, we will create two box plots to we will visualize the relationship between our newly added `hour` and `month` features and energy consumption. These visualizations allow us to view the distribution of energy consumption throughout the hours of the day, and throughout the months of the year.

    
![png](visualizations/bchydro-forecast_20_0.png)

![png](visualizations/bchydro-forecast_21_1.png)
    


---
## Lag Features
Adding lag features to our dataframe allows our model to make more robust predictions, as it effectively gives our model historical context by including historical energy consumption values as features. We will add three lag features, corresponding to one, two, and three years, or 364, 728, and 1092 days in the past respectively. 


```python
target_map = df["energy_kWh"].to_dict()
```


```python
# Define function to add lag features
def add_lags(df):
    df["lag1"] = (df.index - pd.Timedelta("364 days")).map(target_map)
    df["lag2"] = (df.index - pd.Timedelta("728 days")).map(target_map)
    df["lag3"] = (df.index - pd.Timedelta("1092 days")).map(target_map)
    return df
```


```python
add_lags(df)
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
      <th>energy_kWh</th>
      <th>hour</th>
      <th>dayofweek</th>
      <th>quarter</th>
      <th>month</th>
      <th>year</th>
      <th>dayofyear</th>
      <th>lag1</th>
      <th>lag2</th>
      <th>lag3</th>
    </tr>
    <tr>
      <th>datetime</th>
      <th></th>
      <th></th>
      <th></th>
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
      <th>2012-06-01 01:00:00</th>
      <td>1.011</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>6</td>
      <td>2012</td>
      <td>153</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2012-06-01 02:00:00</th>
      <td>0.451</td>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>6</td>
      <td>2012</td>
      <td>153</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2012-06-01 03:00:00</th>
      <td>0.505</td>
      <td>3</td>
      <td>4</td>
      <td>2</td>
      <td>6</td>
      <td>2012</td>
      <td>153</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2012-06-01 04:00:00</th>
      <td>0.441</td>
      <td>4</td>
      <td>4</td>
      <td>2</td>
      <td>6</td>
      <td>2012</td>
      <td>153</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2012-06-01 05:00:00</th>
      <td>0.468</td>
      <td>5</td>
      <td>4</td>
      <td>2</td>
      <td>6</td>
      <td>2012</td>
      <td>153</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
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
      <th>2020-05-20 19:00:00</th>
      <td>0.270</td>
      <td>19</td>
      <td>2</td>
      <td>2</td>
      <td>5</td>
      <td>2020</td>
      <td>141</td>
      <td>0.32</td>
      <td>0.76</td>
      <td>0.49</td>
    </tr>
    <tr>
      <th>2020-05-20 20:00:00</th>
      <td>0.280</td>
      <td>20</td>
      <td>2</td>
      <td>2</td>
      <td>5</td>
      <td>2020</td>
      <td>141</td>
      <td>0.46</td>
      <td>0.63</td>
      <td>0.35</td>
    </tr>
    <tr>
      <th>2020-05-20 21:00:00</th>
      <td>0.280</td>
      <td>21</td>
      <td>2</td>
      <td>2</td>
      <td>5</td>
      <td>2020</td>
      <td>141</td>
      <td>0.52</td>
      <td>0.97</td>
      <td>0.53</td>
    </tr>
    <tr>
      <th>2020-05-20 22:00:00</th>
      <td>0.190</td>
      <td>22</td>
      <td>2</td>
      <td>2</td>
      <td>5</td>
      <td>2020</td>
      <td>141</td>
      <td>0.23</td>
      <td>0.48</td>
      <td>0.17</td>
    </tr>
    <tr>
      <th>2020-05-20 23:00:00</th>
      <td>0.130</td>
      <td>23</td>
      <td>2</td>
      <td>2</td>
      <td>5</td>
      <td>2020</td>
      <td>141</td>
      <td>0.21</td>
      <td>0.18</td>
      <td>0.16</td>
    </tr>
  </tbody>
</table>
<p>69844 rows × 10 columns</p>
</div>



### Training Model with Cross-Validation
We perform 5-fold cross validation on the data, scoring the model using the root mean squared error.


```python
tss = TimeSeriesSplit(n_splits=5, test_size=24 * 365 * 1, gap=24)
df = df.sort_index()


fold = 0
preds = []
scores = []
for train_idx, val_idx in tss.split(df):
    train = df.iloc[train_idx]
    test = df.iloc[val_idx]

    train = make_features(train)
    test = make_features(test)

    feats = [
        "dayofyear",
        "hour",
        "dayofweek",
        "quarter",
        "month",
        "year",
        "lag1",
        "lag2",
        "lag3",
    ]
    target = "energy_kWh"

    X_train = train[feats]
    y_train = train[target]

    X_test = test[feats]
    y_test = test[target]

    reg = xgb.XGBRegressor(
        base_score=0.5,
        booster="gbtree",
        n_estimators=1000,
        early_stopping_rounds=50,
        objective="reg:squarederror",
        max_depth=3,
        learning_rate=0.01,
    )
    reg.fit(
        X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=100
    )

    y_pred = reg.predict(X_test)
    preds.append(y_pred)
    score = np.sqrt(mean_squared_error(y_test, y_pred))
    scores.append(score)
```

    [0]	validation_0-rmse:0.90952	validation_1-rmse:0.86021
    [100]	validation_0-rmse:0.63756	validation_1-rmse:0.64752
    [200]	validation_0-rmse:0.58337	validation_1-rmse:0.62611
    [276]	validation_0-rmse:0.57205	validation_1-rmse:0.62725
    [0]	validation_0-rmse:0.89714	validation_1-rmse:0.45189
    [49]	validation_0-rmse:0.72342	validation_1-rmse:0.53797
    [0]	validation_0-rmse:0.82746	validation_1-rmse:0.42008
    [100]	validation_0-rmse:0.62206	validation_1-rmse:0.39629
    [200]	validation_0-rmse:0.57644	validation_1-rmse:0.38614
    [300]	validation_0-rmse:0.56001	validation_1-rmse:0.38561
    [303]	validation_0-rmse:0.55970	validation_1-rmse:0.38577
    [0]	validation_0-rmse:0.77423	validation_1-rmse:0.40882
    [100]	validation_0-rmse:0.58577	validation_1-rmse:0.38471
    [200]	validation_0-rmse:0.54776	validation_1-rmse:0.38016
    [300]	validation_0-rmse:0.53332	validation_1-rmse:0.37519
    [400]	validation_0-rmse:0.52689	validation_1-rmse:0.37273
    [500]	validation_0-rmse:0.52346	validation_1-rmse:0.37226
    [600]	validation_0-rmse:0.52099	validation_1-rmse:0.37206
    [688]	validation_0-rmse:0.51933	validation_1-rmse:0.37212
    [0]	validation_0-rmse:0.73317	validation_1-rmse:0.59580
    [53]	validation_0-rmse:0.60880	validation_1-rmse:0.59848
    


```python
print(f"Mean Score: {np.mean(scores):0.4f}")
print(f"Score by fold: {scores}")
```

    Mean Score: 0.4860
    Score by fold: [0.6258560172322173, 0.4518899047123867, 0.3844881159145239, 0.37203722712345405, 0.5956534320383671]
    

---
## Future Predictions


```python
# Retrain on all data
df = make_features(df)

feats = [
    "dayofyear",
    "hour",
    "dayofweek",
    "quarter",
    "month",
    "year",
    "lag1",
    "lag2",
    "lag3",
]
target = "energy_kWh"

X_all = df[feats]
y_all = df[target]

reg = xgb.XGBRegressor(
    base_score=0.5,
    booster="gbtree",
    # Adjust n_estimators to accountu for overfitting
    n_estimators=500,
    objective="reg:linear",
    max_depth=3,
    learning_rate=0.01,
)
reg.fit(X_all, y_all, eval_set=[(X_all, y_all)], verbose=100)
```

    [0]	validation_0-rmse:0.71739
    [100]	validation_0-rmse:0.56710
    

    c:\Users\noah8\AppData\Local\Python\Python3.11\Lib\site-packages\xgboost\core.py:160: UserWarning: [14:14:22] WARNING: C:\buildkite-agent\builds\buildkite-windows-cpu-autoscaling-group-i-0b3782d1791676daf-1\xgboost\xgboost-ci-windows\src\objective\regression_obj.cu:209: reg:linear is now deprecated in favor of reg:squarederror.
      warnings.warn(smsg, UserWarning)
    

    [200]	validation_0-rmse:0.53156
    [300]	validation_0-rmse:0.51762
    [400]	validation_0-rmse:0.51017
    [499]	validation_0-rmse:0.50613
    




<style>#sk-container-id-1 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-1 {
  color: var(--sklearn-color-text);
}

#sk-container-id-1 pre {
  padding: 0;
}

#sk-container-id-1 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-1 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-1 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-1 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-1 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-1 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-1 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-1 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-1 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-1 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-1 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-1 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-1 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-1 div.sk-label label.sk-toggleable__label,
#sk-container-id-1 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-1 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-1 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-1 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-1 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-1 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-1 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-1 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-1 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-1 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>XGBRegressor(base_score=0.5, booster=&#x27;gbtree&#x27;, callbacks=None,
             colsample_bylevel=None, colsample_bynode=None,
             colsample_bytree=None, device=None, early_stopping_rounds=None,
             enable_categorical=False, eval_metric=None, feature_types=None,
             gamma=None, grow_policy=None, importance_type=None,
             interaction_constraints=None, learning_rate=0.01, max_bin=None,
             max_cat_threshold=None, max_cat_to_onehot=None,
             max_delta_step=None, max_depth=3, max_leaves=None,
             min_child_weight=None, missing=nan, monotone_constraints=None,
             multi_strategy=None, n_estimators=500, n_jobs=None,
             num_parallel_tree=None, objective=&#x27;reg:linear&#x27;, ...)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;XGBRegressor<span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>XGBRegressor(base_score=0.5, booster=&#x27;gbtree&#x27;, callbacks=None,
             colsample_bylevel=None, colsample_bynode=None,
             colsample_bytree=None, device=None, early_stopping_rounds=None,
             enable_categorical=False, eval_metric=None, feature_types=None,
             gamma=None, grow_policy=None, importance_type=None,
             interaction_constraints=None, learning_rate=0.01, max_bin=None,
             max_cat_threshold=None, max_cat_to_onehot=None,
             max_delta_step=None, max_depth=3, max_leaves=None,
             min_child_weight=None, missing=nan, monotone_constraints=None,
             multi_strategy=None, n_estimators=500, n_jobs=None,
             num_parallel_tree=None, objective=&#x27;reg:linear&#x27;, ...)</pre></div> </div></div></div></div>




```python
# Latest day in our data
df.index.max()
```




    Timestamp('2020-05-20 23:00:00')




```python
# Create future dataframe
future = pd.date_range("2020-05-20 23:00:00", "2021-05-20 3:020:00", freq="1h")
future_df = pd.DataFrame(index=future)
future_df["isFuture"] = True
df["isFuture"] = False
df_and_future = pd.concat([df, future_df])
df_and_future = make_features(df_and_future)
df_and_future = add_lags(df_and_future)

future_w_features = df_and_future.query("isFuture").copy()
```


```python
future_w_features["pred"] = reg.predict(future_w_features[feats])
future_w_features["pred"].plot(figsize=(10, 5), ms=1, lw=1, title="Predicted Energy Consumption")

plt.ylabel("Energy Consumption (kWh)")
plt.xlabel("Date")
plt.set_title("Energy Consumption Forecast: Week of July 13, 2020");
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    Cell In[23], line 6
          4 plt.ylabel("Energy Consumption (kWh)")
          5 plt.xlabel("Date")
    ----> 6 plt.set_title("Energy Consumption Forecast: Week of July 13, 2020");
    

    AttributeError: module 'matplotlib.pyplot' has no attribute 'set_title'



    
![png](bchydro-forecast_files/bchydro-forecast_33_1.png)
    


### Estimated Future Average Weekly Energy Consumption
Using the future prediction data, we will estimate the average weekly energy consumption for future weeks.

#### Seasonal Consumption Comparison
Suppose we are interested in comparing the energy consumption for two given weeks, six months apart; one in the summer, and one in the winter. Let's arbitrarily compare the weeks of July 13, 2020 and January 11, 2021. 


```python
# Week of July 13, 2020
jul12_cons = future_w_features[
    (future_w_features["dayofyear"] >= 195) & (future_w_features["dayofyear"] <= 201)
]["pred"].sum()

# Week of January 11, 2021
jan11_cons = future_w_features[
    (future_w_features["dayofyear"] >= 11) & (future_w_features["dayofyear"] <= 17)
]["pred"].sum()

print("Average Energy Consumption:")
print(f"Week of July 13, 2020: {jul12_cons:0.4f} kW")
print(f"Week of January 11, 2020: {jan11_cons:0.4f} kW")
```

    Average Energy Consumption:
    Week of July 13, 2020: 87.0480 kW
    Week of January 11, 2020: 176.8319 kW
    

#### Visualizing Future Prediction
We can also visualize these two weeks of estimated values to get an idea of how the average energy consumption varies by day of the week. First, let's view the week of July 13, 2020:


```python
# Min, max x-axis tick label
dates = [13, 20]
    
# Plotting predicted values for the week of July 13, 2020
ax = (
    future_w_features["pred"]
    .loc[
        (future_w_features["dayofyear"] >= 195)
        & (future_w_features["dayofyear"] <= 201)
    ]
    .plot(figsize=(15, 5), title="Week of Predictions")
)

# Set plot attributes
ax.set_xticklabels(dates)
ax.set_ylabel("Energy Consumption (kWh)")
ax.set_xlabel("Date (July, 2020)")
ax.set_title("Energy Consumption Forecast: Week of July 13, 2020");
```

    C:\Users\noah8\AppData\Local\Temp\ipykernel_15408\2466733936.py:12: UserWarning: set_ticklabels() should only be used with a fixed number of ticks, i.e. after set_ticks() or using a FixedLocator.
      ax.set_xticklabels(dates)
    


    
![png](bchydro-forecast_files/bchydro-forecast_37_1.png)
    


Now, let's visualize the predicted energy consumption for the week of January 11, 2021:


```python
# Min, max x-axis tick labels
dates = [11, 18]

# Plotting predicted values for the week of January 11, 2021
ax = (
    future_w_features["pred"]
    .loc[
        (future_w_features["dayofyear"] >= 11) & (future_w_features["dayofyear"] <= 17)
    ]
    .plot(figsize=(15, 5), color="#48a739")
)

# Set plot attributes
ax.set_xticklabels(dates)
ax.set_ylabel("Energy Consumption (kWh)")
ax.set_xlabel("Date (January, 2021)")
ax.set_title("Energy Consumption Forecast: Week of January 11, 2021");
```

    C:\Users\noah8\AppData\Local\Temp\ipykernel_15408\812824184.py:11: UserWarning: set_ticklabels() should only be used with a fixed number of ticks, i.e. after set_ticks() or using a FixedLocator.
      ax.set_xticklabels(dates)
    




    Text(0.5, 1.0, 'Energy Consumption Forecast: Week of January 11, 2021')




    
![png](bchydro-forecast_files/bchydro-forecast_39_2.png)
    


## Discussion
The predictions/estimates provided by the model developed in this project has the potential to offer BC Hydro several key benefits. Such insights allow for ***more efficient resource management*** by optimizing power generation and distribution, ensuring a reliable supply without overproduction or shortages. 

Forecasting also supports better infrastructure planning and maintenance, reducing costs and preventing outages. Additionally, it allows for ***improved integration of renewable energy sources***, balancing demand with variable supply from sources like wind or solar. This, in addition to other predictive insights, help BC Hydro manage peak demand more effectively, offering customers cost-saving initiatives and contributing to a more sustainable energy system.

