# app.py

from flask import Flask,request,jsonify
from flask_cors import CORS

# loading the data from the csv file to apandas dataframe

app = Flask(__name__)
cors = CORS(app, resources={r"*": {"origins": "*"}})
from recommendwithhist import recommend_movieswithhistory
from recommendwithdesc import recommend_movies_with_desc
from recommend_normal import recommend_movies
# Define a route for the home page
@app.route('/')

def hello_world():
 # You can assign a unique ID to each user
  
    return recommend_movieswithhistory(request.args.get('username'),request.args.get('movie'))
    
@app.route('/des')
def test():
  # You can assign a unique ID to each user
    print(request.args.get('desc'))
    return recommend_movies_with_desc([request.args.get('desc')])
@app.route('/search')
def normal():
    
    return recommend_movies(request.args.get('movie'))
# Run the app if the script is executed directly
if __name__ == '__main__':
    app.run(debug=True)
