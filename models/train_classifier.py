import sys

if len(sys.argv) == 3:
    # import libraries
    import pandas as pd
    import numpy as np
    from sqlalchemy import create_engine

    import re

    import nltk
    nltk.download(['punkt',
                   'wordnet',
                   'averaged_perceptron_tagger',
                   'stopwords'],
                   quiet=True)

    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords

#    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report
    from sklearn.model_selection import GridSearchCV
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline, FeatureUnion
#    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

    from sklearn.multioutput import MultiOutputClassifier

    import pickle

def load_data(database_filepath):
    # Connect to the database
    engine = create_engine('sqlite:///{}'.format(database_filepath))

    # Create a dataframe
    df = pd.read_sql("SELECT * FROM data", engine)

    # Create feature and target variables
    X = df["message"].values
    Y = df[df.columns[4:]].to_numpy()

    # Keep the names of the target columns
    category_names = df.columns[4:]
    return X, Y, category_names

def tokenize(text):
    # normalize the text
    text = re.sub(r"[^a-zA-Z0-9?-]", " ", text.lower())

    # tokenize the text + remove stopwords
    tokens = [token
                for token in word_tokenize(text)
                if token not in stopwords.words("english")]
    lemmatizer = WordNetLemmatizer()

    # lemmatize
    lemmed = [lemmatizer.lemmatize(token).lower().strip() for token in tokens]

    return lemmed


def build_model():
    # Create a pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize))
        , ('tfidf', TfidfTransformer())
        , ('classifier', MultiOutputClassifier(KNeighborsClassifier()))
        ])

    # Specify the parameters for the GridSearchCV
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.75, 1.0),
        'vect__max_features': (None, 5000),
        'tfidf__use_idf': (True, False),
        'classifier__estimator__n_neighbors': [5],
        'classifier__estimator__leaf_size': [25, 30]
    }

    cv = GridSearchCV(pipeline, param_grid = parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):

    # Get the values predicted by model for the test set
    Y_pred = model.predict(X_test)

    # Create the classification report as one table
    ## For each category create a dataframe using classification_report
    ## and append them together
    df_classification_report = pd.concat(
        [( pd.DataFrame(
            classification_report(Y_test[:,i],
                                  Y_pred[:,i],
                                  labels = [0, 1],
                                  target_names = [col + " : 0", col + " : 1"],
                                  output_dict = True,
                                  zero_division = 0))
            .iloc[:,:2].transpose().round(2))
         for i, col in enumerate(category_names)])

    # Support should be an integer
    df_classification_report.support = df_classification_report.support.astype(int)

    # Print out the table
    pd.set_option('display.max_rows', None)
    print(df_classification_report)

    pass


def save_model(model, model_filepath):

    # Save the best model
    pickle.dump(model.best_estimator_, open(model_filepath,'wb'))

    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
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
