
import requests
import pandas as pd

def get_data():
    training_label_url = 'https://raw.githubusercontent.com/kyledufrane/Tanzanian_Water_Wells/main/data/Training_set_labels.csv'
    training_values_url = 'https://raw.githubusercontent.com/kyledufrane/Tanzanian_Water_Wells/main/data/Training_set_values.csv'

    tr_lbl_df = pd.read_csv(training_label_url)
    tr_val_df = pd.read_csv(training_values_url)

    df = pd.merge(tr_val_df, tr_lbl_df, on='id', how='outer')

    def get_elevation(lat, long):
        query = ('https://api.open-elevation.com/api/v1/lookup'f'?locations={lat},{long}')
        r = requests.get(query).json()
        elevation = pd.json_normalize(r, 'results')['elevation'].values[0]
        return elevation

    df['elevation'] = 0

    df['elevation'] = df.apply(lambda x: get_elevation(x['latitude'], x['longitude']), axis=1)

    df.to_csv('data/data_with_elevation.csv')

if __name__ == '__main__':
    get_data()
