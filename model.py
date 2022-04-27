import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import seaborn as sns

from math import sqrt
from scipy import stats
import sklearn.preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Don't bother with warning me...
import warnings
warnings.filterwarnings("ignore")
# See everything
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)
# Set default seaborn style
plt.rc('figure', figsize=(12, 10))
plt.style.use('fivethirtyeight')

########################################### Correlation ###########################################

def IsPass_correlation(df):
    df.corr()['IsPass'].sort_values().plot(kind='barh')
    plt.title('Correlation of Features to IsPass')
    return

######################################### Question Vizzes #########################################

def q1_viz(df):
    plt.title('Passing and Rushing Has Appeared to Remain Relatively Constant')
    sns.countplot(data = df, x= 'SeasonYear', hue = 'PlayType')
    plt.xlabel('Season Year')
    plt.ylabel('Number of Plays')
    plt.ylim(0,19_999)
    plt.legend(['Run', 'Pass'], prop ={'size':18})
    plt.show()
    return

def q3_viz(df):
    plt.title('As Down Increases, Rushing Count Decreases at a Faster Rate Than Passing')
    sns.countplot(data = df, x= 'Down', hue = 'IsPass')
    plt.xlabel('Down')
    plt.ylabel('Number of Plays')
    plt.legend(['Run', 'Pass'], prop ={'size':18})
    plt.show()
    return

def q4_viz(df):
    plt.title('Pecentage of Passes Thrown in Each Quarter')
    ax = sns.barplot(x=df.Quarter, y=df.IsPass, data= df, hue ='YTG_bins')
    plt.xlabel('Quarter')
    plt.ylabel('Percentage Pass')
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.0%}'))
    plt.ylim(0,1.19)
    plt.show()
    return

def q2_viz(df):
    # Make an easy data frame that lists the findings in order 
    # of Seasons with the highest yardage produced
    return pd.DataFrame(df.groupby('SeasonYear')['Yards']
                .sum().sort_values(ascending = False))
    

########################################## Question Stats #########################################

def q3_stats(df):
    x = df['Down']
    y = df['IsPass']
    alpha = 0.05
    r, p = stats.pearsonr(x,y)
    null_hypothesis = "there is no linear relationship \nbetween down and the play resulting in a pass."
    print('=============================================================')
    print('r =', r)
    print('p =', p)
    print('-------------------------------------------------------------')
    if p < alpha:
        print("We reject the hypothesis that", null_hypothesis)
    else:
        print("We fail to reject the null hypothesis")
    print('=============================================================')
    return 

def q4_stats(df):
    # Set our alpha
    alpha = 0.05
    # Crosstab Attrition vs Business Travel
    observed = pd.crosstab(df.IsPass, df.Quarter)
    # .chi2_contingency returns 4 different values
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    null_hypothesis = "IsPass and Quarter are independent."
    print('=================================================================')
    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p}')
    print('-----------------------------------------------------------------')
    if p < alpha:
        print("We reject the hypothesis that", null_hypothesis)
    else:
        print("We fail to reject the null hypothesis")
    print('=================================================================') 
    return

 ######################################### Modeling Functions #########################################

# Create cluster for seconds left in the game and yards to go for a 1st down
def get_togo_cluster(train, train_scaled, validate, validate_scaled, test, test_scaled):
    # Set target variables
    target_var = 'IsPass'

    #Make copies of train, validate, and test
    train_scaled = train_scaled.copy()
    validate_scaled = validate_scaled.copy()
    test_scaled = test_scaled.copy()

    # create X_train by dropping the target variable 
    X_train = train_scaled.drop(columns=[target_var])
    # create y_train by keeping only the target variable.
    y_train = train[[target_var]]

    # create X_validate by dropping the target variable 
    X_validate = validate_scaled.drop(columns=[target_var])
    # create y_validate by keeping only the target variable.
    y_validate = validate[[target_var]]

    # create X_test by dropping the target variable 
    X_test = test_scaled.drop(columns=[target_var])
    # create y_test by keeping only the target variable.
    y_test = test[[target_var]]

    # Get the cluster of the togo
    # Scale Needed data for cluster
    # Group and name 
    cluster_vars = ['ToGo', 'SecondsLeft']
    cluster_name = 'togo_cluster'
    # Create Scaler
    scaler = MinMaxScaler()
    # Note that we only call .fit with the training data,
    # but we use .transform to apply the scaling to all the data splits.
    scaler.fit(X_train[cluster_vars])
    # Transform
    X_train_scaled = scaler.transform(X_train[cluster_vars])
    X_validate_scaled = scaler.transform(X_validate[cluster_vars])
    X_test_scaled = scaler.transform(X_test[cluster_vars]) 
    # Create KMeans
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X_train_scaled)
    kmeans.predict(X_train_scaled)
    # Store the predicted cluster back into our original dataframe.
    train['togo_cluster'] = kmeans.predict(X_train_scaled)
    train.togo_cluster = train.togo_cluster.astype('category')
    validate['togo_cluster'] = kmeans.predict(X_validate_scaled)
    validate.togo_cluster = validate.togo_cluster.astype('category')
    test['togo_cluster'] = kmeans.predict(X_test_scaled)
    test.togo_cluster = test.togo_cluster.astype('category')
    return

# Set baseline based on the mode of IsPass which = 1
def baseline(df):
    df['baseline'] = 1
    baseline_accuracy = (df.baseline == df.IsPass).mean()
    print(f'Baseline is {(baseline_accuracy):.3%}')
    return

# Set x and y's based on features
def set_x_and_y(train, validate, test):
    #Define features used for the model
    x_cols =['togo_cluster', 'Down', 'YardLine']

    # Create x & y version of train, where y is a series with just the target variable 
    # and X are all the features. 
    # Train
    x_train = train[x_cols]
    y_train = train[['IsPass']]
    # Validate
    x_validate = validate[x_cols]
    y_validate = validate[['IsPass']]
    # Test
    x_test = test[x_cols]
    y_test = test[['IsPass']]
    return x_train, y_train, x_validate, y_validate, x_test, y_test

def model_accuracy(x_train, y_train, x_validate, y_validate, x_test, y_test):
    #Define features used for the model
    x_cols =['togo_cluster', 'Down', 'YardLine']
    #Create Logistic Regression Model
    logit = LogisticRegression(random_state=123)
    # Fit the model
    logit.fit(x_train, y_train)
    # Establish weights
    weights = logit.coef_.flatten()
    # Establish intercept
    pd.DataFrame(weights, x_cols).reset_index().rename(columns={'index': 'x_cols', 0: 'weight'})
    logit = LogisticRegression(C=1, random_state=123)
    logit.fit(x_train, y_train)

    # Create a Decision Tree model and set Tree max depth
    tree = DecisionTreeClassifier(max_depth = 6)
    # Fit the model
    tree.fit(x_train,y_train.IsPass)

    # Create a Random Forest model and set the number of trees and the max depth of 6 
    # based on loop used to find best performing k-value
    # Create the model with max depth of 16
    rf = RandomForestClassifier(max_depth=16,min_samples_leaf=16,random_state=1349)
    # Fit the model
    rf.fit(x_train, y_train)  

    # Create a KNN model and set the number of neighbors to be used at 5
    knn = KNeighborsClassifier(n_neighbors=5)
    # Fit the model
    knn.fit(x_train,y_train)

    # Print the accuracy of each model
    print('==================================================================')
    # Accuracy on train for  Logistic Regression:
    print(f'Accuracy of Logistic Regression on the training set is {(logit.score(x_train, y_train)):.3%}')
    # Accurcy on validate for Logistic Regression:
    print(f'Accuracy of Logistic Regression on the validation set is {(logit.score(x_validate, y_validate)):.3%}')
    print('------------------------------------------------------------------')
    # Accuracy on train for the Decision Tree:
    print(f'Accuracy of Decision Tree Classifier on the training set is {(tree.score(x_train, y_train)):.3%}')
    # Accuracy on validate for the Decision Tree:
    print(f'Accuracy of Decision Tree Classifier on the validation set is {(tree.score(x_validate, y_validate)):.3%}')
    print('------------------------------------------------------------------')
    # Accuracy on train for the Random Forest:
    print(f'Accuracy of Random Forest on the training set is {(rf.score(x_train, y_train)):.3%}')
    # Accurcy on validate for the Random Forest:
    print(f'Accuracy of Random Forest on the validation set is {(rf.score(x_validate, y_validate)):.3%}')
    print('------------------------------------------------------------------')
    # Accuracy on train for  KNN:
    print(f'Accuracy of KNN on the training set is {(knn.score(x_train, y_train)):.3%}')
    # Accurcy on validate for KNN:
    print(f'Accuracy of KNN on the validation set is {(knn.score(x_validate, y_validate)):.3%}')
    print('==================================================================')
    return

def decision_tree_best_on_test(x_test, y_test, df):
    #Create the model
    tree = DecisionTreeClassifier(max_depth=6, random_state=123)
    # Fit the model
    tree.fit(x_test, y_test)
    # Evaluate the model
    # Accuracy on train for the Decision Tree:
    df['baseline'] = 1
    baseline_accuracy = (df.baseline == df.IsPass).mean()
    print(f'Baseline Accuracy of Decision Tree Classifier is {(baseline_accuracy):.3%}')
    # Accurcy on validate for the Decision Tree:
    print(f'Accuracy of Decision Tree Classifier on the test set is {(tree.score(x_test, y_test)):.3%}')
    # By how much
    print(f'Accuracy of Decision Tree Classifier on the test set is {(tree.score(x_test, y_test) - baseline_accuracy):.3%}') 
