from flask import Flask, render_template, request, url_for
import pickle

# Loading files
clf = pickle.load(open('nb_amz_2.pkl', 'rb'))
cv = pickle.load(open('amz_transform_2.pkl', 'rb'))

# Initiate flask
app = Flask(__name__)

# Route for home page
@app.route('/')
def home():
    return render_template('Tweets_Predict_Page.html')
    

# Route for predictor page
@app.route('/Predict', methods = ['POST'])
def Predict():
    
    if request.method == 'POST':
        tweet = request.form['message']
        data = [tweet]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('Tweets_Predict_Page.html', prediction = my_prediction)



if __name__ == '__main__':
    app.run(debug = False)