import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import datetime
import requests
import os

# Don't bother with warning me...
import warnings
warnings.filterwarnings("ignore")
# See everything
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)
# Set default seaborn style
plt.rc('figure', figsize=(10, 7))
plt.style.use('fivethirtyeight')


######################################### Acquire Data #########################################

def acquire_play_by_play():
    """
    Acquire play-by-play data from multiple csv's and update each file
    acquired from the directory where they
    """
    # Define the needed file where csv's will be located
    csv_files = os.listdir('plays_file')
    dataframes = []

    # Read in each file with pandas
    for file in csv_files:
        if not file.endswith('.csv'):
            continue
        print(file, 'acquired')
        df = pd.read_csv('plays_file/'+file)
        df['file'] = file
        dataframes.append(df)
    # Combine them all together
    df = pd.concat(dataframes)
    return df

######################################### Data Cleaning #########################################

def prep_play_by_play(df):
    '''
    This function drops any columns and rows deemed unneeded for the 
    scope of our project. It then sets out index to GameDate in order 
    to use time series analysis.
    '''
    # Define the columns we want to drop    
    drop_cols = ['GameId','Unnamed: 10','Unnamed: 12', 'TeamWin','Unnamed: 16', 'Unnamed: 17', 'NextScore', 
       'IsIncomplete', 'IsTouchdown','PassType', 'IsSack', 'IsChallenge', 'IsChallengeReversed', 'Description',
       'Challenger', 'IsMeasurement','IsPenalty', 'IsTwoPointConversion', 'IsTwoPointConversionSuccessful',
       'RushDirection','IsPenaltyAccepted', 'PenaltyTeam', 'IsNoPlay', 'PenaltyType','PenaltyYards', 'file',
       'IsRush']
    # Drop columns identified as not needed
    df = df.drop(columns=drop_cols)
    # Drop rows for kick formations
    df = df[df.Formation != 'PUNT']
    df = df[df.Formation != 'FIELD GOAL']
    # Drop rows where play type was not pass or rush
    df = df[df.PlayType != 'KICK OFF']
    df = df[df.PlayType != 'TIMEOUT']
    df = df[df.PlayType != 'SACK']
    df = df[df.PlayType != 'EXTRA POINT']
    df = df[df.PlayType != 'SCRAMBLE']
    df = df[df.PlayType != 'NO PLAY']
    df = df[df.PlayType != 'QB KNEEL']
    df = df[df.PlayType != 'TWO-POINT CONVERSION']
    df = df[df.PlayType != 'EXCEPTION']
    df = df[df.PlayType != 'FUMBLES']
    df = df[df.PlayType != 'PUNT']
    df = df[df.PlayType != 'CLOCK STOP']
    df = df[df.PlayType != 'PENALTY']
    df = df[df.PlayType != 'FIELD GOAL']
    df = df[df.Down != 0]
    # Create bine for yards to go
    df['YTG_bins'] = pd.cut(df.ToGo, [0,1, 3, 11, 44], labels=['inches', 'short', 'medium', 'long'])
    # Establish helpful time categories
    df['QuarterSeconds'] = ((df.Quarter*15)*60)
    df['ClockSeconds'] = ((df.Minute * 60)+ df.Second)
    df['SecondsLeft'] = (3600- (df.QuarterSeconds - df.ClockSeconds))
    # Set GameDate to datetime and index it
    # df['GameDate'] = pd.to_datetime(df['GameDate'])
    # df.set_index('GameDate', inplace=True)
    # df.sort_index(inplace=True)
    return df

######################################### Split the Data ###########################################

def split_data(df):
    '''
    This function takes in the dataframe and target variable name as arguments and then
    splits the dataframe into train (56%), validate (24%), & test (20%)
    It will return a list containing the following dataframes: train (for exploration), 
    X_train, X_validate, X_test, y_train, y_validate, y_test
    '''
    # split df into train_validate (80%) and test (20%)
    train_validate, test = train_test_split(df, test_size=.20, random_state=123)
    # split train_validate into train(70% of 80% = 56%) and validate (30% of 80% = 24%)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)

#     #Make copies of train, validate, and test
#     train = train.copy()
#     validate = validate.copy()
#     test = test.copy()
    
#     # create X_train by dropping the target variable 
#     X_train = train.drop(columns=[target_var])
#     # create y_train by keeping only the target variable.
#     y_train = train[[target_var]]

#     # create X_validate by dropping the target variable 
#     X_validate = validate.drop(columns=[target_var])
#     # create y_validate by keeping only the target variable.
#     y_validate = validate[[target_var]]

#     # create X_test by dropping the target variable 
#     X_test = test.drop(columns=[target_var])
#     # create y_test by keeping only the target variable.
#     y_test = test[[target_var]]
    df_total = (train.shape[0]+validate.shape[0]+test.shape[0])
    print(f'Data frame sizes are as follow: \nTrain = {train.shape[0]} \nValidate = {validate.shape[0]} \nTest = {test.shape[0]} \nTotal dataframe = {df_total}')
    return train, validate, test

######################################## Min-Max Scaler ############################################

def min_max_scaler(train, validate, test):
    '''
    Scales the 3 data splits using the MinMaxScaler()
    takes in the train, validate, and test data splits and returns their scaled counterparts.
    If return_scaler is true, the scaler object will be returned as well.
    '''
    # Make the scaler
    scaler = MinMaxScaler()
    # List columns that need to be scaled
    cols = train[['IsPass', 'Quarter', 'SecondsLeft', 'ToGo', 'ToGo', 'Down']].columns.tolist()
    # Make a copy of original train, validate, and test
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    # Use/fit the Scaler
    train_scaled[cols] = scaler.fit_transform(train[cols])
    validate_scaled[cols] = scaler.fit_transform(validate[cols])
    test_scaled[cols] = scaler.fit_transform(test[cols])

    return train_scaled, validate_scaled, test_scaled

############################### Feature Engineering and Modeling ###################################
############################# Find The Best K Value For Clustering #################################

def inertia(df, feature1, feature2, r1, r2):
    cols = [feature1, feature2]
    X = df[cols]
    
    inertias = {}
    
    for k in range(r1, r2):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        inertias[k] = kmeans.inertia_
    
    pd.Series(inertias).plot(xlabel='k', ylabel='Inertia', figsize=(9, 7)).plot(marker='x')
    plt.grid()
    return

##################################### Show Clusters/Centroids #######################################

def cluster(df, feature1, feature2, k):
    X = df[[feature1, feature2]]

    kmeans = KMeans(n_clusters=k).fit(X)
    
    df['cluster'] = kmeans.labels_
    df.cluster = df.cluster.astype('category')
    
    df['cluster'] = kmeans.predict(X)

    centroids = pd.DataFrame(kmeans.cluster_centers_, columns=X.columns)

    df.groupby('cluster')[feature1, feature2].mean()
    
    plt.figure(figsize=(9, 7))
    
    for cluster, subset in df.groupby('cluster'):
        plt.scatter(subset[feature1], subset[feature2],  label='cluster ' + str(cluster), 
                    alpha=.6)
    
    centroids.plot.scatter(x=feature1, y=feature2, c='black', marker='x', s=100, ax=plt.gca(),
                           label='centroid')
    
    plt.legend()
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.title('Visualizing Cluster Centers')

    return

#####################################################################################################