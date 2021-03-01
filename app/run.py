import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

import nltk
nltk.download(['punkt',
               'wordnet',
               'averaged_perceptron_tagger',
               'stopwords'],
                quiet=True)
from nltk.corpus import stopwords

import re

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
# from sklearn.externals import joblib as extjoblib
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

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

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('data', engine)
import os
os.getcwd()
# load model
model = joblib.load("../models/classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    category_freq_array = df.iloc[:,4:].sum().sort_values(ascending=False)
    category_names = category_freq_array.index
    category_counts = category_freq_array.values

    # create visuals
    graphs = [

            # the first graph: genre

        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },

            # the second graph - the category
            {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': -30,
                    'automargin': True
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
