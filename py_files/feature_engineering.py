import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from sklearn.impute import KNNImputer


def filter_column_strings(threshold, dataframe, column, scorer=fuzz.ratio, drop: bool = False):
    while len(filter_strings(threshold, dataframe, column, scorer=scorer, drop=drop)) > 0:
        filter_strings(threshold, dataframe, column, scorer=scorer, drop=drop)


def filter_strings(threshold, df, col, scorer, drop: bool = False, ):
    filter_list_ = df[f'{col}'].unique().tolist()

    score_sort = [(X,) + i for X in filter_list_ for i in
                  process.extract(X, filter_list_, scorer=scorer)]

    scores = pd.DataFrame(score_sort, columns=[f'{col}', f'{col}_match', 'score']).sort_values('score', ascending=False)
    scores = scores[scores[f'{col}'] != scores[f'{col}_match']]
    scores[f'unique_{col}'] = np.minimum(scores[f'{col}'], scores[f'{col}_match']).str.lower()
    scores = scores[(scores.score >= threshold)]

    if drop:
        wanted_installers = scores[f'{col}'].tolist()
        for name in wanted_installers:
            df[f'{col}'].replace(name, scores[scores[f'{col}'] == name][f'unique_{col}'].iloc[0], inplace=True)

    return scores


def knn_impute(df, test_df, col, nan_val_replacements, n_neighbors):

    df[col] = df[col].replace(nan_val_replacements, np.nan)
    test_df[col] = test_df[col].replace(nan_val_replacements, np.nan)

    knn_impute = KNNImputer(n_neighbors=n_neighbors, weights='distance')

    X = df[['latitude', 'longitude', col]]
    X_test = test_df[['latitude', 'longitude', col]]

    imputed = knn_impute.fit_transform(X)
    test_imputed = knn_impute.transform(X_test)

    df[col] = pd.DataFrame(imputed)[2]
    test_df[col] = pd.DataFrame(test_imputed)[2]

    return df, test_df

def cat_label_pct(df, test_df, col):

    temp_df = df.groupby(col)['status_group'] \
        .value_counts(normalize=True) \
        .to_frame() \
        .rename(columns={'status_group':'percentage'}) \
        .reset_index()

    temp_df['status_group'] = col + '_' + (temp_df['status_group'] \
                                          .str \
                                          .replace(' ', '_')) \
                                          .astype(str)
    temp_df = temp_df.pivot_table('percentage', col, 'status_group') \
        .reset_index()

    df = pd.merge(df, temp_df, on=col)
    test_df = pd.merge(test_df, temp_df, on=col, how='left')


    return df, test_df