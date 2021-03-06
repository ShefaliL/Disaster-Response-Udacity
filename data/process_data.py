import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    load_data
    Load data from csv files and merge to a single pandas dataframe

    Input:
    messages_filepath filepath to messages csv file
    categories_filepath filespath to categories csv file

    Returns:
    df dataframe merging categories and messages
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on = 'id', how = 'inner')
    return df


def clean_data(df):
    '''
    clean_data
    Clean data in the dataframe, convert column from string to numeric

    Input:
    df dataframe to be cleaned

    Returns:
    df cleaned dataframe
    '''
    categories = df['categories'].str.split(pat = ';',expand = True)
    row = categories.iloc[0]
    category_colnames =list(map(lambda x: x[:-2],row))
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1:]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    df.drop(columns=['categories'],axis = 1, inplace = True)
    df = pd.concat([df,categories], join='outer', axis=1)
    df.drop_duplicates(inplace = True)
    return df
    


def save_data(df, database_filename):
    '''
    save_data
    Save the dataframe to an sql table

    Input:
    df dataframe to be cleaned
    database_filename database file name where ETL_message needs to be saved

    Returns:
    df cleaned dataframe
    '''
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('ETL_message', engine, index=False, if_exists='replace')


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