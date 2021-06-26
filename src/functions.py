# --------------------------------------------------------------
# Define library for all functions within this notebook
# --------------------------------------------------------------

# Import libaries needed for EDA and visualizations
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Import Pickle to saved files giving us the ability to only run each model once.
import pickle

# Import needed SKLearn libraries for modeling, imputing, and pipelines
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, plot_confusion_matrix
from xgboost import XGBClassifier

# Import YellowBrick to easily plot ROC AUC curves on multiclass algorithms
from yellowbrick.classifier import ROCAUC


def replace_other(dataframe, columns):
    '''Replaces null values with string other
    
       dataframe = pandasdataframe
       
       columns = column string ** must be inserted as 'column_name' 
       
       '''
    
    for column in columns:
        dataframe[column] = dataframe[column].replace(np.nan, 'Other', regex = True)
    
    return None


def replace_most_frequent(dataframe, columns):
    '''Replaces null values with column mode
    
    dataframe = pandas dataframe 
    
    columns = column string ** must be inserted as 'column_name'
    
    '''

    for column in columns:
        dataframe[column].fillna(dataframe[column]\
                        .value_counts().index[0], inplace = True)
    return None


def get_totals(dataframe, filter_column, filter_groupby):

        '''
        Function takes in dataframe, column, and column groupby arguement.

        1. get_totals will calculate the sum of the variables
        within a column and return a new column with the 
        sum of their total occurances in the dataframe
        
        2. get_totals will calulate the percentage of the 
        values column vs the total values

        dataframe = pandas dataframe
        filter_column = column to filter by
        filter_groupby = groupby column to filter by

        '''

        df_new = pd.DataFrame(dataframe.groupby(filter_groupby)[filter_column].value_counts())
        df_new[f'{filter_groupby}_values'] = df_new[filter_column]
        df_new.drop(filter_column, axis = 1, inplace = True)
        df_new.reset_index(inplace = True)

        types = set()

        for idx, value in enumerate(df_new[f'{filter_groupby}_values']):
            for type_ in df_new[filter_column]:
                types.add(type_)
            
        total_values = {}
            
        for value in types:
            total_values[value] = df_new[df_new[filter_column] == value][f'{filter_groupby}_values'].sum()

        df_new[f'{filter_groupby}_total_values'] = df_new[filter_column].map(total_values)

        df_new[f'{filter_groupby}_percentage'] = df_new[f'{filter_groupby}_values'] / df_new[f'{filter_groupby}_total_values']
            
        return df_new

    
def plot_barh_def(column, status_group):
    
    '''
    Function takes in column and status group arguement and 
    returns a horizontal bar chart
    
    Note: This function can only be used if the percentage_dict 
    dictionary is loaded in the environment. See above code. 
        
        File is located in the source_code folder
    
    column = pandas dataframe column
        
        must be entered as a string
        
    status_group must be one of the following:
    
        'functional'
        'non functional'
        'functional needs repair'
    
    
    '''
    pickle_in = open('saved_code/percentage_dict.pickle', 'rb')

    percentage_dict = pickle.load(pickle_in)  


    sns.set_theme(style='darkgrid')
    
    graph = percentage_dict[column]
    graph_funct = graph[graph['status_group'] == status_group]
    graph_funct = graph_funct.sort_values('status_group_percentage', ascending = False)

    fig, ax = plt.subplots(figsize=(10,10))
    ax.barh(graph_funct[column], graph_funct['status_group_percentage'])
    plt.xlabel('Percentage of {} wells'.format(status_group))
    plt.ylabel(column)
    plt.yticks(rotation=30)
    plt.title('Percentage of {} wells by {}'.format(status_group, column))
    plt.savefig('saved_objects/{}_{}'.format(column, status_group));
    
    
    
def cat_models(columns, X_train, y_train):
    
    ''' Function takes in categorical values, OneHotEncodes, and 
    returns all metrics associated to the RandomForestClassifier
    
    Note: RFC parameters =
                            random_state = 42
                            class_weight = 'balanced'
                            n_jobs = -1 
    Arguements:
    
    columns - must be categorical
    
    X_train
    
    y_train
    
    
    Output:
    
    Recall Score
    
    Precision Score
    
    F1 Score
    
    Cross Val Score = cv = 3, scoring = 'recall_macro'
    
    Confusion Matrix
    
    ROCAUC - * Predictors can be multiclass *
    '''
    
    
    
    # Initiallize OHE
    ohe = OneHotEncoder(drop = 'first')
    
    # Create wanted dataframe
    df_feat = X_train[columns]
    
    # Fit transform categorical columns
    df_feat_enc = ohe.fit_transform(df_feat)
    
    # Instantiate classifier
    rfc = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
    
    #Fit Classifier
    rfc.fit(df_feat_enc, y_train)
    
    # Model Score
    score = rfc.score(df_feat_enc, y_train)
    
    # Predict X_train
    yhat = rfc.predict(df_feat_enc)
    
    # Model Recall
    recall = recall_score(y_train, yhat, average='macro')
    print('Recall Score:', recall)
    
    # Model Precision
    precision = precision_score(y_train, yhat, average='macro')
    print('Precision Score:', precision)
    
    # Model F1
    f1 = f1_score(y_train, yhat, average='macro')
    print('F1 Score:', f1)
    
    # Cross-validate
    cross_val = cross_val_score(rfc, df_feat_enc, y_train, cv=3, scoring='recall_macro')
    print('Cross Val Score:', cross_val)
    print('Mean Cross Val Score:', np.mean(cross_val))
    
    # Plot Confusion Matrix
    
    fig, ax = plt.subplots(figsize=(12,5), ncols=2)
    plot_confusion_matrix(rfc, df_feat_enc, y_train, ax=ax[0], xticks_rotation=30)
    plt.title('Confusion Matrix')
    ax[0].grid(None)
    
    # ROC AUC via Yellowbrick
    visualizer = ROCAUC(rfc)
    visualizer.fit(df_feat_enc, y_train)
    visualizer.score(df_feat_enc, y_train)
    plt.legend()
    plt.tight_layout()

    return None



def pipe_rfc(num_feats, cat_feats, X_train, y_train):
    ''' This function takes in numerical and categorical arguements along with 
    X_train and y_train. Function passes arguements through a pipeline with 
    StandardScaler and OneHotEncoder. 
    
    Arguements:
    
    num_feats - numerical columns
    
    cat_feats - catergorical columns
    
    X_train
    
    y_train
    
    
    Output:
    
    Recall Score
    
    Precision Score
    
    F1 Score
    
    Cross Val Score - cv=3, scoring = 'recall_macro'
    
    Confusion Matrix
    
    ROCAUC - *can be multiclass*
    
    '''
    
    
    # Create Numberic Transformer
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())])
    
    # Create Categorical transformer
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    # Create Preprossor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_feats),
            ('cat', categorical_transformer, cat_feats)])

    # Create Pipeline
    rfc = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', RandomForestClassifier(random_state = 42, class_weight= 'balanced', n_jobs = -1))])

    # Fit Model
    rfc.fit(X_train, y_train)
            
    # Model Score
    score = rfc.score(X_train, y_train)
    
    # Predict X_train
    yhat = rfc.predict(X_train)
    
    # Model Recall
    recall = recall_score(y_train, yhat, average='macro')
    print('Recall Score:', recall)
    
    # Model Precision
    precision = precision_score(y_train, yhat, average='macro')
    print('Precision Score:', precision)
    
    # Model F1
    f1 = f1_score(y_train, yhat, average='macro')
    print('F1 Score:', f1)
    
    # Cross-validate
    cross_val = cross_val_score(rfc, X_train, y_train, cv=3, scoring='recall_macro')
    print('Cross Val Score:', cross_val)
    print('Mean Cross Val Score:', np.mean(cross_val))
    
    # Plot Confusion Matrix
    
    fig, ax = plt.subplots(figsize=(12,5), ncols=2)
    plot_confusion_matrix(rfc, X_train, y_train, ax=ax[0], xticks_rotation=30)
    plt.title('Confusion Matrix')
    ax[0].grid(None)
    
    # ROC AUC via Yellowbrick
    visualizer = ROCAUC(rfc)
    visualizer.fit(X_train, y_train)
    visualizer.score(X_train, y_train)
    plt.legend()
    plt.tight_layout()
    
    return None


def scoring(X_test, y_test, model):
    '''This function takes in X_test, y_test, and instantiated model. 
    
    Arguements:
    
    X_test
    
    y_test
    
    model
    
    Output:
    
    Recall Score
    
    Precision Score
    
    F1 Score
    
    Cross Val Score - cv = 3, scoring = 'recall_macro'
    
    Confusion Matrix
    
    ROCAUC - *can be multiclass*
     
    '''
    
    # Predict X_train
    yhat = model.predict(X_test)
    
    # Model Recall
    recall = recall_score(y_test, yhat, average='macro')
    print('Recall Score:', recall)
    
    # Model Precision
    precision = precision_score(y_test, yhat, average='macro')
    print('Precision Score:', precision)
    
    # Model F1
    f1 = f1_score(y_test, yhat, average='macro')
    print('F1 Score:', f1)
    
    # Cross-validate
    cross_val = cross_val_score(model, X_test, y_test, cv=3, scoring='recall_macro')
    print('Cross Val Score:', cross_val)
    print('Mean Cross Val Score:', np.mean(cross_val))
    
    # Plot Confusion Matrix
    
    fig, ax = plt.subplots(figsize=(12,5), ncols=2)
    plot_confusion_matrix(model, X_test, y_test, ax=ax[0], xticks_rotation=30)
    plt.title('Confusion Matrix')
    ax[0].grid(None)
    
    # ROC AUC via Yellowbrick
    visualizer = ROCAUC(model)
    visualizer.fit(X_test, y_test)
    visualizer.score(X_test, y_test)
    plt.legend()
    plt.tight_layout()