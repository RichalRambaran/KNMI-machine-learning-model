import numpy as np
import pickle
from flask import Flask, request, render_template


# initializing app
app = Flask(__name__)

# retrieve dumped model
# data_folder = Path("model")
# file_to_open = data_folder / "model_date_temperature.pkl"
model = pickle.load(open("model_date_temperature.pkl", "rb"))


@app.route("/")
# default page of the app
def home():
    return render_template("index.html")


# retrieve values from form when button is clicked
@app.route("/", methods=["POST"])
def predict_temperature():
    # retrieve values from form
    int_features = [int(x) for x in request.form.values()]
    print(int_features)

    # save values
    values = [np.array(int_features)]
    print(values)

    # send values to model en save in variable:
    v = model.predict(values)
    print(v)

    # format temperature
    outcome = np.round((v/10), 1)

    # values are sent back to html page in variable answer
    return render_template("index.html", answer="The predicted temperature is: " + format(outcome) + " °C")


if __name__ == "__main__":
    app.run(debug=True)
