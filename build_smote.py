import pandas as pd
from imblearn.over_sampling import SMOTENC, ADASYN
from sklearn.preprocessing import OneHotEncoder


def build_data():
    df_training_labels = pd.read_csv('data/Training_set_labels.csv')
    df_training_values = pd.read_csv('data/Training_set_values.csv')
    df = df_training_values.merge(df_training_labels, on='id')
    df.drop('id', axis=1, inplace=True)

    df.fillna('Unknown', inplace=True)

    drop_columns = [
        'date_recorded',
        'wpt_name',
        'recorded_by',
        'scheme_name',
        'waterpoint_type_group',
        'source_class',
        'source',
        'quantity_group',
        'quality_group',
        'payment_type',
        'management_group',
        'extraction_type',
        'extraction_type_group',
    ]

    df.drop(drop_columns, axis=1, inplace=True)

    df['construction_year'] = df['construction_year'].replace(0, df['construction_year'].median())

    funder_mask = df['funder'].map(df['funder'].value_counts()) < 800
    df['funder'] = df['funder'].mask(funder_mask, 'other')

    df[['public_meeting', 'permit']] = df[['public_meeting', 'permit']].astype(bool)

    drop_cols = ['longitude', 'population', 'permit', 'region_code', 'gps_height', 'latitude']
    df.drop(drop_cols, axis=1, inplace=True)

    X = df.drop(['status_group'], axis=1)
    y = df['status_group']

    idx_val = []

    for idx, col in enumerate(X.select_dtypes(['object', 'bool']).columns.to_list()):
        for idx_, col_ in enumerate(X.columns.to_list()):
            if col == col_:
                idx_val.append(idx_)

    X_smote, y_smote = SMOTENC(categorical_features=idx_val, n_jobs=-1, k_neighbors=5).fit_resample(X, y)
    print('SMOTENC Completed')
    smote_data = pd.concat([X_smote, y_smote], axis=1)
    smote_data.to_csv('data/smote_data.csv')
    print('Exported SMOTENC')

if __name__ == "__main__":
    build_data()
