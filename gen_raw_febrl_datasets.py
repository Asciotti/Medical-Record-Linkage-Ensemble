import recordlinkage as rl, pandas as pd, numpy as np
from recordlinkage.datasets import load_febrl3, load_febrl4
from tqdm import tqdm

def clean_data(df):
    df['date_of_birth']  = pd.to_datetime(df['date_of_birth'], errors = 'coerce')
    df['day'] = df['date_of_birth'].dt.strftime('%d')
    df['month'] = df['date_of_birth'].dt.strftime('%m')
    df['year'] = df['date_of_birth'].dt.strftime('%Y')

    df['postcode'] = df['postcode'].fillna('0000')
    df['postcode'] = df['postcode'].astype(int)

    df['street_number'] = df['street_number'].fillna('0')
    df['street_number'] = df['street_number'].astype(int)

    df = df.drop(["soc_sec_id",  "date_of_birth"], axis=1)

    for col in ["surname", "given_name", "address_1", "address_2", "day", "month"]:
        df[col] = df[col].fillna('')
        df[col] = df[col].astype(str)

    df['match_id'] = [-1]*len(df)

    all_fields = df.columns.values.tolist()
    print("All fields:", all_fields)
    df.head()

    return df

def find_links(df, true_links):
    for i in tqdm(range(len(true_links))):
        k0 = true_links[i][0]
        k1 = true_links[i][1]
        df.at[k0, "match_id"] = i
        df.at[k1, "match_id"] = i
    
    return df

def gen_raw_train_data():
    df, true_links = load_febrl3(return_links=True)
    df = clean_data(df)
    df = find_links(df, true_links)
    df.to_csv("febrl_UNSW_train.csv", index=True)

def gen_raw_test_data():
    df_a, df_b, true_links = load_febrl4(return_links=True)
    df = df_a.append(df_b) # Package splits up two chunks of data for some reason 
    df = clean_data(df)
    df = find_links(df, true_links)
    df.to_csv("febrl_UNSW_test.csv", index=True)


if __name__ == '__main__':
    gen_raw_train_data()
    gen_ra w_test_data()
