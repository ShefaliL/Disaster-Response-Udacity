# Disaster-Response-Udacity

Link to the classifier.pickel file: https://drive.google.com/file/d/1GHmwZ7QuvfZO2sSsK0cPhVpazTyQEw8E/view?usp=sharing

# ETL Pipeline
The first part of your data pipeline is the Extract, Transform, and Load process. 
Here, I read the dataset, cleaned it  and then stored it in a SQLite database. I did the data cleaning with the help of pandas. 
To load the data into an SQLite database, I have used the pandas dataframe .to_sql() method, which you can use with an SQLAlchemy engine.

The cleaned code is in the final ETL script, process_data.py.

# Machine Learning Pipeline
For the machine learning portion, the data is splitted into training set and a testing set. 
Then, a machine learning pipeline has been created that uses NLTK, as well as scikit-learn's Pipeline and GridSearchCV to output a final model that uses the message column to predict classifications for 36 categories (multi-output classification). 
Finally, exported the model to a pickle file. 
After completing the notebook, I have included the final machine learning code in train_classifier.py.

# Data Pipelines: Python Scripts
After you complete the notebooks for the ETL and machine learning pipeline, you'll need to transfer your work into Python scripts, process_data.py and train_classifier.py. If someone in the future comes with a revised or new dataset of messages, they should be able to easily create a new model just by running your code. 
These Python scripts should be able to run with additional arguments specifying the files used for the data and model.

# Flask App
In the last step, you'll display your results in a Flask web app. 
We have provided a workspace for you with starter files. You will need to upload your database file and pkl file with your model.

This is the part of the project that allows for the most creativity. 
So if you are comfortable with html, css, and javascript, feel free to make the web app as elaborate as you would like.

In the starter files, you will see that the web app already works and displays a visualization. 
You'll just have to modify the file paths to your database and pickled model file as needed.

There is one other change that you are required to make. 
We've provided code for a simple data visualization. Your job will be to create two additional data visualizations in your web app based on data you extract from the SQLite database. You can modify and copy the code we provided in the starter files to make the visualizations.

Github and Code Quality
Throughout the process, make sure to push your code and comments to Github so that you will not repeat your work and you can keep track of the changes you've made. 
This will also help you keep your code modular and well documented. Make sure to include effective comments and docstrings. 
These software engineering practices will improve your communication and collaboration in the future when you work within a team.
