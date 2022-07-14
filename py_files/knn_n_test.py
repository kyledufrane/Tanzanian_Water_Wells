import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.impute import KNNImputer
import numpy as np

df = pd.read_csv('data/cleaned_data.csv')


def cat_label_pct(df, col):
    temp_df = df.groupby(col)['status_group'] \
        .value_counts(normalize=True) \
        .to_frame() \
        .rename(columns={'status_group': 'percentage'}) \
        .reset_index()

    temp_df['status_group'] = col + '_' + (temp_df['status_group'] \
                                           .str \
                                           .replace(' ', '_')) \
        .astype(str)
    temp_df = temp_df.pivot_table('percentage', col, 'status_group') \
        .reset_index()

    df = pd.merge(df, temp_df, on=col)

    return df


def test_knnimpute(df, n_neighbors):
    wanted_cols = ['basin', 'subvillage', 'region', 'region_code', 'district_code', 'lga', 'ward', 'funder',
                   'installer', 'region_code', 'district_code']
    unwanted_cols = ['subvillage', 'lga', 'ward', 'district_code', 'region_code']

    for i in wanted_cols:
        if i not in unwanted_cols:
            df = cat_label_pct(df, i)
            df.drop(i, axis=1, inplace=True)

    df['construction_year'] = df['construction_year'].replace(0, 2000)
    df['construction_year'] = 2022 - df['construction_year']
    df = df.rename(columns={'construction_year': 'well_age'})

    df['urban'] = df['lga'].apply(lambda x: 1 if 'urban' in x else 0)
    df.drop('lga', axis=1, inplace=True)

    unwanted_cols = ['date_recorded', 'wpt_name', 'subvillage', 'status_group', 'scheme_name', 'recorded_by','funder',\
                     'installer', 'ward']

    for col in df.select_dtypes('object').columns:
        if col not in unwanted_cols:
            df = cat_label_pct(df, col)
            df.drop(col, axis=1, inplace=True)

    scaler = StandardScaler()
    df[['latitude', 'longitude']] = scaler.fit_transform(df[['latitude', 'longitude']])

    df.population = df.population.replace([0, 1], np.nan)
    knn_impute = KNNImputer(n_neighbors=n_neighbors, weights='distance')
    X = df[['latitude', 'longitude', 'population']]

    imputed = knn_impute.fit_transform(X)
    df['population'] = pd.DataFrame(imputed)[2]

    df.gps_height = df.gps_height.replace(0, np.nan)
    knn_impute = KNNImputer(n_neighbors=n_neighbors, weights='distance')
    X = df[['latitude', 'longitude', 'gps_height']]

    imputed = knn_impute.fit_transform(X)
    df['gps_height'] = pd.DataFrame(imputed)[2]

    df.amount_tsh = df.amount_tsh.replace(0, np.nan)
    knn_impute = KNNImputer(n_neighbors=n_neighbors, weights='distance')
    X = df[['latitude', 'longitude', 'amount_tsh']]

    imputed = knn_impute.fit_transform(X)
    df['amount_tsh'] = pd.DataFrame(imputed)[2]

    df = df[df.status_group.notna()].copy()
    df.fillna(0.0, inplace=True)

    corr_feats = set()
    corr = df.corr()

    for i in range(len(corr.columns)):
        for j in range(i):
            if abs(corr.iloc[i, j]) > 0.80:
                colname = corr.columns[i]
                corr_feats.add(colname)

    df_ = df.drop(corr_feats, axis=1)

    unwanted_cols = ['date_recorded', 'subvillage', 'recorded_by', 'wpt_name', 'Unnamed: 0', 'id']
    df_.drop(unwanted_cols, axis=1, inplace=True)

    X = df_.drop(['status_group'], axis=1).select_dtypes(['int', 'float'])
    y = df_['status_group']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    rf = RandomForestClassifier(random_state=42, n_jobs=-1)

    rf.fit(X_train, y_train)

    y_hat = rf.predict(X_test)

    print('n_neighbors:', n_neighbors, 'Accuracy Score:', accuracy_score(y_test, y_hat))

