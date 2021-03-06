import pandas as pd
import time


def one_hot_encode(df, columns=[], inplace=False):
    if type(columns) != list:
        columns = [columns]

    new_df = df if inplace else df.loc[:]

    for feat in columns:
        if feat in df.columns:
            new_feat = pd.get_dummies(df.loc[:, [feat]])
            new_df[new_feat.columns] = new_feat
            new_df.drop(feat, axis=1, inplace=True)

    return new_df


def timeit(f):
    def timed(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print('took %2.4f sec' % (te-ts))
        return result

    return timed


def replace_values(df, columns, value_pairs, inplace=False):
    if type(columns) is not list:
        columns = [columns]

    if not inplace:
        df = pd.DataFrame(df.loc[:])

    for col in columns:
        for key, val in value_pairs:
            df.loc[df[col] == key, col] = val
    if not inplace:
        return df
