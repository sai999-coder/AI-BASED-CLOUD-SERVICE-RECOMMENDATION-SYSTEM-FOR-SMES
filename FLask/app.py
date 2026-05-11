from flask import Flask, render_template, request
import pickle
import plotly
import plotly.graph_objects as go
import json

app = Flask(__name__)

# Load model
with open("model.pkl", "rb") as f:
    reg_model, clf_model, le_dict = pickle.load(f)

def encode_input(data):
    cols = ["budget", "storage", "scalability", "service"]
    encoded = []
    
    for i, col in enumerate(cols):
        encoded.append(le_dict[col].transform([data[i]])[0])
    
    return [encoded]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    budget = request.form['budget']
    storage = request.form['storage']
    scalability = request.form['scalability']
    service = request.form['service']

    input_data = encode_input([budget, storage, scalability, service])

    # Predictions
    reg_pred = reg_model.predict(input_data)[0]
    clf_pred = clf_model.predict(input_data)[0]

    provider = le_dict["recommended_provider"].inverse_transform([clf_pred])[0]

    price = round(reg_pred[0], 2)
    speed = round(reg_pred[1], 2)
    storage_space = round(reg_pred[2], 2)

    # Plotly chart
    fig = go.Figure()
    fig.add_trace(go.Bar(x=["Price", "Speed", "Storage"], 
                         y=[price, speed, storage_space]))

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template("result.html",
                           provider=provider,
                           price=price,
                           speed=speed,
                           storage_space=storage_space,
                           graphJSON=graphJSON)

if __name__ == '__main__':
    app.run(debug=True)
