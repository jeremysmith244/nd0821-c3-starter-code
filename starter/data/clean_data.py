import pandas as pd

if __name__ == '__main__':

    df = pd.read_csv('census.csv')

    df.columns = [x.strip() for x in df.columns]

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.strip()

    df.to_csv('clean_census.csv')