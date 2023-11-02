# Download the California Housing Price from Kaggle: https://www.kaggle.com/datasets/camnugent/california-housing-prices
# and save as dataset_housing.csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error


def prepare_X(train, val, test, filled_value):
    train = train.copy()
    val = val.copy()
    test = test.copy()

    train['total_bedrooms'] = train['total_bedrooms'].fillna(filled_value).copy()
    val['total_bedrooms'] = val['total_bedrooms'].fillna(filled_value).copy()
    test['total_bedrooms'] = test['total_bedrooms'].fillna(filled_value).copy()

    dv = DictVectorizer(sparse=False)
    train_dict = train.to_dict(orient='records')
    train_X = dv.fit_transform(train_dict)
    val_dict = val.to_dict(orient='records')
    val_X = dv.transform(val_dict)
    test_dict = test.to_dict(orient='records')
    test_X = dv.transform(test_dict)

    return train_X, val_X, test_X

def split_df(df, random_stat):
    df = df.copy()
    full_train_set, test_set = train_test_split(df, test_size=0.2, random_state=random_stat)
    train_set, val_set = train_test_split(full_train_set, test_size=0.25, random_state=random_stat)

    y_train = train_set.median_house_value.values
    y_val = val_set.median_house_value.values
    y_test = test_set.median_house_value.values

    del train_set['median_house_value']
    del val_set['median_house_value']
    del test_set['median_house_value']

    return train_set, val_set, test_set, y_train, y_val, y_test

if __name__ =="__main__":
    # Data preparation
    df = pd.read_csv('data_housing.csv')

    df.columns = df.columns.str.lower().str.replace(' ', '_')

    strings = list(df.dtypes[df.dtypes == 'object'].index)
    for col in strings:
        df[col] = df[col].str.lower().str.replace(' ', '_')
    print(df.dtypes)

    # EDA
    for col in df.columns:
        print(col)
        print(df[col].unique())
        print(df[col].nunique())
        print()

    # Distribution of median_house_value
    sns.histplot(df.median_house_value, bins=50)
    plt.savefig('media_house_value_hist.png')
    plt.close()

    # plot mean_house_value with ocean_proximity <1h_ocean or inland
    sns.histplot(df.median_house_value[df.ocean_proximity.isin(['<1h_ocean', 'inland'])])
    plt.savefig('media_house_value_hist_proximity.png')
    plt.close()

    # 1. Use subset with ocean_proximity <1h_ocean or inland
    print(df.ocean_proximity.value_counts())
    df = df[df.ocean_proximity.isin(['<1h_ocean', 'inland'])]

    # Use subset with the following variables
    cols = [
    'latitude',
    'longitude',
    'housing_median_age',
    'total_rooms',
    'total_bedrooms',
    'population',
    'households',
    'median_income',
    'median_house_value'
]

    df = df[cols]

    # 2. Check features with missing values
    print(df.isnull().sum())

    # 3. Median (50% percentile) for variable 'population'
    print(df.population.median())

    # 4. Prepare the dataset:
    ## Shuffle the dataset (the filtered one you created above), use seed 42.
    ## Split your data in train/val/test sets, with 60%/20%/20% distribution.
    ## Apply the log transformation to the median_house_value variable using the np.log1p() function.
    df.median_house_value = np.log1p(df.median_house_value)
    full_train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
    train_set, val_set = train_test_split(full_train_set, test_size=0.25, random_state=42)

    y_train = train_set.median_house_value
    y_val = val_set.median_house_value
    y_test = test_set.median_house_value

    del train_set['median_house_value']
    del val_set['median_house_value']
    del test_set['median_house_value']

    # 5. We need to deal with missing values for the column from 1).
    ## We have two options: fill it with 0 or with the mean of this variable.
    ## Try both options. For each, train a linear regression model without regularization.
    ## For computing the mean, use the training only!
    ## Use the validation dataset to evaluate the models and compare the RMSE of each option.
    ## Round the RMSE scores to 4 decimal digits
    ## Which option gives better RMSE?

    # Fill the missing values with mean
    mean = train_set.total_bedrooms.mean()
    X_train, X_val, X_test = prepare_X(train_set, val_set, test_set, mean)
    model = LinearRegression()
    model.fit(X_train, y_train)

    pred_val = model.predict(X_val)
    rmse = mean_squared_error(pred_val, y_val, squared=False)
    print('RMSE ', rmse.round(4))  # Fill with mean: 0.34076415314137004

    # Fill the missing values with 0
    X_train, X_val, X_test = prepare_X(train_set, val_set, test_set, 0)
    model = LinearRegression()
    model.fit(X_train, y_train)

    pred_val = model.predict(X_val)
    rmse = mean_squared_error(pred_val, y_val, squared=False)
    print('RMSE ', rmse.round(4))  # Fill with 0: 0.3412529566695128

    #5. Train regularized linear regression.
    # Fill the NAs with mean.
    # Try different values of r from this list: [0, 0.000001, 0.0001, 0.001, 0.01, 0.1, 1, 5, 10].
    # Use RMSE to evaluate the model on the validation dataset.
    # Round the RMSE scores to 2 decimal digits.
    # Which r gives the best RMSE?
    for r in [0, 0.000001, 0.0001, 0.001, 0.01, 0.1, 1, 5, 10]:
        model = Ridge(alpha=r)
        model.fit(X_train, y_train)
        pred_val = model.predict(X_val)
        rmse = mean_squared_error(pred_val, y_val, squared=False)
        print('r = %f, rmse = %f' %(r, rmse))

    #6. Useseed 42 for splitting the data. Let's find out how selecting the seed influences our score.
    #Try different seed values: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].
    #For each seed, do the train/validation/test split with 60%/20%/20% distribution.
    #Fill the missing values with 0 and train a model without regularization.
    #For each seed, evaluate the model on the validation dataset and collect the RMSE scores.
    #What's the standard deviation of all the scores? To compute the standard deviation, use np.std.
    #Round the result to 3 decimal digits (round(std, 3))
    scores = []
    for s in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        train_set, val_set, test_set, y_train, y_val, y_test = split_df(df, s)
        X_train, X_val, X_test = prepare_X(train_set, val_set, test_set, 0)
        model = LinearRegression()
        model.fit(X_train, y_train)
        pred_val = model.predict(X_val)
        score = mean_squared_error(pred_val, y_val, squared=False)
        scores.append(score)
        print('s = %f, rmse = %f' %(s, score))
    rmse_s = np.std(scores)
    print('std: ', rmse_s.round(3))

    # 7. Split the dataset like previously, use seed 9.
    # Combine train and validation datasets.
    # Fill the missing values with 0 and train a model with r=0.001.
    # What's the RMSE on the test dataset?
    train_set, val_set, test_set, y_train, y_val, y_test = split_df(df, 9)
    X_train, X_val, X_test = prepare_X(train_set, val_set, test_set, 0)
    X_train_full = np.concatenate((X_train, X_val), axis=0)
    y_train_full = np.concatenate((y_train, y_val), axis=0)

    model = LinearRegression()
    model.fit(X_train_full, y_train_full)
    pred_test = model.predict(X_test)
    rmse = mean_squared_error(pred_test, y_test, squared=False)
    print('rmse: ', rmse)

