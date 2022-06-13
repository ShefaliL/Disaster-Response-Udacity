import sys
import pickle
import os.path
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet'])

import numpy as np
import re
import pandas as pd
import sys
import os
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.externals import joblib
import nltk
nltk.download('stopwords')


def load_data(database_filepath):
    '''
    load_data
    Load data from given database specified by the database_filepath

    Input:
    database_filepath filepath to database

    Returns:
    X storing messages column
    y storing the target column
    category_names all category column names
    '''
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table('ETL_message', engine)
    X = df.message
    y = df.iloc[:,4:]
    category_names = list(df.columns[4:])
    
    return X, y, category_names


def tokenize(text):
    '''
    tokenize
    Tokenizing the given text

    Input:
    text input text

    Returns:
    tokens returning tokens derived fromthe given text
    '''
    # Making all the text as lowercase
    text = text.lower()
    # Removing the punctuations using re module
    text = re.sub(r'[^\w\s]','', text)
    # Removing numbers
    text = re.sub(r'[0-9]+','', text)
    # Tokenizing the words
    tokens = word_tokenize(text)
    # Removing Stop words
    stop_words = set(stopwords.words('english'))
    tokens = [i for i in tokens if not i in stop_words]
    # Lemmatizing the tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return tokens 


def build_model():
    '''
    build_model
    Building our ML model

    Input:

    Returns:
    pipeline our ML pipeline model
    '''
    pipeline = Pipeline([
        ('cnt_vect', CountVectorizer(tokenizer = tokenize)),
        ('tf_idf', TfidfTransformer()),
        ('mo_clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
#     parameters = {
# #     'cnt_vect__max_features' : [None, 10000, 50000],
# #     'tf_idf__use_idf' : [True, False],
#     'mo_clf__estimator__n_estimators' : [100,200,300], 
#     'mo_clf__estimator__min_samples_split' : [2,3,4]
#     }

#     clf_grid = GridSearchCV(pipeline, param_grid = parameters, verbose = 2, n_jobs = -1)
    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    evaluate_model
    evaluating our model based on test data

    Input:
    model  our ML model
    X_test  X test
    Y_test Y test
    category_names column names

    Returns:

    '''
    test_pred = model.predict(X_test)
        # print the metrics
    for i, col in enumerate(category_names):
        print('{} category metrics: '.format(col))
        print(classification_report(Y_test.iloc[:,i], test_pred[:,i]))
    


def save_model(model, model_filepath):
    '''
    save_model
    saving our model to a pickle file

    Input:
    model  our ML model
    model_filepath  file path to our save our model

    Returns:
    '''
#     n_bytes = 2**31
#     max_bytes = 2**31 - 1
#     data = bytearray(n_bytes)

#     ## write
#     bytes_out = pickle.dumps(model)
#     with open(model_filepath, 'wb') as f_out:
#         for idx in range(0, len(bytes_out), max_bytes):
#             f_out.write(bytes_out[idx:idx+max_bytes])
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print(sys.argv[1:])
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()