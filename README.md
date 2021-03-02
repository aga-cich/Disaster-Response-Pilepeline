# Disaster Response Pipeline Project

### Project Overview

This project is a part of the Udacity's Data Scientist Nanodegree program.

The goal of this project is to build a model that is able to classify disaster messages. The model will be trained on the disaster data from [Figure Eight](https://www.figure-eight.com/). In the dataset there are 36 categories of the messages (e.g. Aid Related, Medical Help, Water, Food, Shelter). Every message can receive multiple labels, making it a multi-label classification problem.

The end product is a web app where you can enter a new message that will get classified as none, one or more of the predefined categories. In the app you will also be able to see some basic stats about the dataset used to train the model.

### Installation

Packages:
* pandas
* numpy
* sys
* json
* sqlalchemy
* flask
* sklearn
* nltk

### Instructions:

#### Run ETL Pipeline

In the project directory run the following command:

```bat
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```

#### Run ML Pipeline

In the project directory run the following command:

```bat
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
```

#### Run the web app

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/ or http://localhost:3001/

### File Descriptions

Disaster_Response_Pipeline
* app
    * templates
        * go.html
        * master.html
    * run.py
* data
    * disaster_message.csv - _dataset with categories_
    * disaster_categories.csv - _dataset with messages_
    * DisasterResponse.db - _output of process_data.py_
    * process_data.py - _ETL pipeline: load, clean, create a SQLite database_
* models
    * (|-- classifier.pkl - _not included in GitHub because the file is too large; output of train_classifier.py_)
    * train_classifier.py - _ML Pipeline: load the data from the database, build and export the final model_
* README
* .gitignore
