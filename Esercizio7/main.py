import os
import pandas as pd 
import numpy as np 
import flask
import pickle
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import column_or_1d
from sklearn.model_selection import train_test_split
from flask import Flask, render_template, request
seed=1234
app=Flask(__name__)


class SonarParser(object):
    def __init__(self,path):

        self.URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"
        self.name = "sonar"
        self.file_name = path
        self.file_path = path
        self.label_col = "60"
        self.X, self.y = self._parse_file()
        self.all = pd.concat([self.X, self.y], axis=1)
        # Metrica Scelta per il test
        self.metric = "accuracy"
        self._print_stats()

    def _parse_file(self,):
        """
            -Read csv data
            -Drop nan values
            -Keep only numeric columns
            -Split to X for features and y for labels
        """
        data = pd.read_csv(self.file_path)
        # rimuove i valori Nan
        data_cleaned = data.dropna()

        X, y = data_cleaned.drop(columns=[self.label_col]), data_cleaned[self.label_col]

        # keep only numeric features
        X = X.loc[:, X.dtypes == np.float64].dropna()

        return X, y

    def save_to_csv(self):
        save_path = os.path.join("..", "..", "data", "interim", self.file_name)
        self.all.to_csv(save_path, index=False)

    def _print_stats(self):
        print("#"*30 + " Start Dataset - " + self.name + " Stats " + "#"*30)
        print("Dataset shape:", self.all.shape)
        print("Counts for each class:")
        print(self.y.value_counts())
        print("Sample of first 5 rows:")
        print(self.all.head(5))
        print("#"*30 + " End Dataset Stats " + "#"*30)

@app.route('/')
def index():
    return flask.render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    if request.method == "POST":
        
        to_predict_list = request.form.to_dict()
        
        model_path= to_predict_list["model"]
        csv_path= to_predict_list["csv"]
        index = int(to_predict_list["index"])
        
        # load csv
        parser= SonarParser(csv_path)

        y_sonar = column_or_1d(parser.y, warn=False)
        X_train, X_test, y_train, y_test = train_test_split(parser.X, y_sonar, test_size=0.20, random_state=seed)

        # Load model

        model = pickle.load(open(model_path, 'rb'))
        prediction = model.predict(np.array(X_test.iloc[index]).reshape(1,-1))
        
        label= y_test[index]

        print(parser.y)
        
        # result = ValuePredictor(to_predict_list)
        # prediction = str(result)
        return render_template("predict.html",prediction=prediction,label=label)
if __name__ =="__main__":

    app.run(debug=True,port="7080")


