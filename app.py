from flask import Flask, render_template, request
import numpy as np
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Training data
X = np.array([
    [1000,10,20],
    [1500,12,25],
    [2000,15,30],
    [2500,18,40],
    [3000,20,50]
])

y = np.array([6,7,8,10,12])

model = LinearRegression()
model.fit(X,y)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():

    size = float(request.form['size'])
    workers = float(request.form['workers'])
    budget = float(request.form['budget'])

    prediction = model.predict([[size,workers,budget]])

    result = round(prediction[0],2)

    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)