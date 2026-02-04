from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained pipeline
model = joblib.load("pipeline.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = {
        "Age": int(request.form["Age"]),
        "Gender": request.form["Gender"],
        "Department": request.form["Department"],
        "Designation": request.form["Designation"],
        "ExperienceYears": int(request.form["ExperienceYears"]),
        "Skillset": request.form.getlist("Skillset"),
        "ProductivityScore": int(request.form["ProductivityScore"]),
        "WorkLocation": request.form["WorkLocation"],
        "EducationLevel": request.form["EducationLevel"],
        "LastPromotionYear": int(request.form["LastPromotionYear"])
    }

    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]

    return render_template(
        "index.html",
        prediction_text=f"Prediction Result: {prediction}",
        form_data=data
    )

if __name__ == "__main__":
    app.run(debug=True)
