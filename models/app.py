from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load model and encoders
model = joblib.load("student_model.pkl")
encoders = joblib.load("encoders.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    gender = request.form["gender"]
    race = request.form["race"]
    education = request.form["education"]
    lunch = request.form["lunch"]
    prep = request.form["prep"]

    math_score = int(request.form["math"])
    reading_score = int(request.form["reading"])
    writing_score = int(request.form["writing"])

    # Encode categorical values
    gender = encoders["gender"].transform([gender])[0]
    race = encoders["race/ethnicity"].transform([race])[0]
    education = encoders["parental level of education"].transform([education])[0]
    lunch = encoders["lunch"].transform([lunch])[0]
    prep = encoders["test preparation course"].transform([prep])[0]

    data = pd.DataFrame([[
        gender,
        race,
        education,
        lunch,
        prep,
        math_score,
        reading_score,
        writing_score
    ]], columns=[
        "gender",
        "race/ethnicity",
        "parental level of education",
        "lunch",
        "test preparation course",
        "math score",
        "reading score",
        "writing score"
    ])

    prediction = model.predict(data)[0]

    result = "Likely to Pass ✅" if prediction == 1 else "Likely to Fail ❌"

    return render_template(
        "index.html",
        prediction=result
    )


if __name__ == "__main__":
    app.run(debug=True)