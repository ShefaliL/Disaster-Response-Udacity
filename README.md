# Disaster-Response-Udacity

Link to the classifier.pickel file: https://drive.google.com/file/d/1GHmwZ7QuvfZO2sSsK0cPhVpazTyQEw8E/view?usp=sharing

# Description
This project was created in conjunction with Figure Eight as part of Udacity's Data Science Nanodegree Program. Pre-labeled tweets and communications from real-life crisis occurrences are included in the dataset.The goal of the project is to create a Natural Language Processing (NLP) model that can categorize communications in real time.
The following essential sections make up this project:
- Building an ETL pipeline to extract data from a source, clean the data, and save it in a SQLite database.
- Create a machine learning pipeline to train a text classification system that can categorize messages into several categories.
- Create a web application that can display model results in real time.

# File Structure

![Screenshot 2022-06-12 222600](https://user-images.githubusercontent.com/76598077/173268591-568a00fd-8d63-4336-a401-e0572f249e8b.png)

# Dependencies
- Python 3.5+
- Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
- Natural Language Process Libraries: NLTK
- SQLlite Database Libraqries: SQLalchemy
- Model Loading and Saving Library: Pickle
- Web App and Data Visualization: Flask, Plotly

# Installing
To clone the git repository: https://github.com/ShefaliL/Disaster-Response-Udacity

# Executing Program:
- You can run the following commands in the project's directory to set up the database, train model and save the model.
-To run ETL pipeline to clean data and store the processed data in the database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_response_db.db
- To run the ML pipeline that loads data from DB, trains classifier and saves the classifier as a pickle file python models/train_classifier.py data/disaster_response_db.db models/classifier.pkl
- Run the following command in the app's directory to run your web app. python run.py
- Go to http://0.0.0.0:3001/

# To create a processed sqlite db
python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
# To train and save a pkl model
python train_classifier.py ../data/DisasterResponse.db classifier.pkl
# To deploy the application locally
python run.py

# Additional Material
- In the data and models folder you can find two jupyter notebook that will help you understand how the model works step by step:
-ETL Preparation Notebook: learn everything about the implemented ETL pipeline
- ML Pipeline Preparation Notebook: look at the Machine Learning Pipeline developed with NLTK and Scikit-Learn

# Important Files
- app/templates/*: templates/html files for web app
- data/process_data.py: Extract Train Load (ETL) pipeline used for data cleaning, feature extraction, and storing data in a SQLite database
- models/train_classifier.py: A machine learning pipeline that loads data, trains a model, and saves the trained model as a .pkl file for later use
- run.py: This file can be used to launch the Flask web app used to classify disaster messages

Authors
Shefali Luley

Acknowledgements
Udacity for providing an amazing Data Science Nanodegree Program
Figure Eight for providing the relevant dataset to train the model

Screenshots



![Web page 1](https://user-images.githubusercontent.com/76598077/173120038-e8f3ce89-d6ce-4fc9-b656-a679958cc91f.png)
![Web page 2](https://user-images.githubusercontent.com/76598077/173120040-483d5af2-b9f2-4bc7-bc08-c828b63c007d.png)
![Web page 3](https://user-images.githubusercontent.com/76598077/173120041-c286efb2-b7f5-4b0f-a226-54b65c0b070c.png)
![Web page 4](https://user-images.githubusercontent.com/76598077/173120042-3f371ec1-b8bf-4ab8-8358-b7199928ae88.png)







