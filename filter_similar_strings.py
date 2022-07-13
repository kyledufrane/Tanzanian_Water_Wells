def filter_strings(threshold, df, column_, drop:bool=False):
    col = column_
    filter_list_ = df[f'{col}'].unique().tolist()

    score_sort = [(X,) + i for X in filter_list_ for i in process.extract(X, filter_list_, scorer=fuzz.token_sort_ratio)]

    scores = pd.DataFrame(score_sort, columns=[f'{col}', f'{col}_match', 'score']).sort_values('score', ascending=False)
    scores = scores[scores[f'{col}'] != scores[f'{col}_match']]
    scores[f'unique_{col}'] = np.minimum(scores[f'{col}'], scores[f'{col}_match']).str.lower()
    scores = scores[(scores.score >= threshold)]

    if drop:
        wanted_installers = scores.installer.tolist()
        for name in wanted_installers:
            df[f'{col}'].replace(name, scores[scores[f'{col}'] == name][f'unique_{col}'].iloc[0], inplace=True)

    return scores
