# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):

    # load messages dataset
    messages = pd.read_csv(messages_filepath, dtype=str)
    categories = pd.read_csv(categories_filepath, dtype=str)

    # merge datasets
    df = (messages.merge(categories,
                     how="outer",
                     on=["id"])
     )
    return df



def clean_data(df):
    # extract "categories" from df
    # split "categories" by ";" into separate columns
    # take the last sign of each cell (which should be 0 or 1)
    # and convert all the values to numbers
    categories = (df['categories'].str.split(';',expand=True)
                    .applymap(lambda x: x[-1])
                 )

    # convert columns from string to numeric
    categories = categories.apply(pd.to_numeric, errors='coerce')

    # select the first row of the categories dataframe
    row = df['categories'][0]

    # use row to extract column names for categories_filepath
    category_colnames = [x.split("-")[0] for x in row.split(';')]

    # rename the columns of `categories`
    categories.columns = category_colnames

    # drop the original categories column from `df`
    df.drop(["categories"], axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df



def save_data(df, database_filename):
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('data', engine, index=False, if_exists='replace')
    pass


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
