from flask import Flask, render_template, request
import joblib
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)

model = joblib.load("output_model/xgb_final_model.joblib")
preprocessor = joblib.load("output_model/preprocessor.joblib")

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    form_data = {}

    if request.method == "POST":
        form_data = {
            "RevolvingUtilization": request.form["RevolvingUtilization"],
            "Age": request.form["Age"],
            "PastDue30_59": request.form["PastDue30_59"],
            "DebtRatio": request.form["DebtRatio"],
            "MonthlyIncome": request.form["MonthlyIncome"],
            "OpenCreditLines": request.form["OpenCreditLines"],
            "Late90": request.form["Late90"],
            "RealEstateLoans": request.form["RealEstateLoans"],
            "PastDue60_89": request.form["PastDue60_89"],
            "Dependents": request.form["Dependents"]
        }

        input_data = {
            "RevolvingUtilizationOfUnsecuredLines": float(form_data["RevolvingUtilization"]),
            "age": int(form_data["Age"]),
            "NumberOfTime30-59DaysPastDueNotWorse": int(form_data["PastDue30_59"]),
            "DebtRatio": float(form_data["DebtRatio"]),
            "MonthlyIncome": float(form_data["MonthlyIncome"]),
            "NumberOfOpenCreditLinesAndLoans": int(form_data["OpenCreditLines"]),
            "NumberOfTimes90DaysLate": int(form_data["Late90"]),
            "NumberRealEstateLoansOrLines": int(form_data["RealEstateLoans"]),
            "NumberOfTime60-89DaysPastDueNotWorse": int(form_data["PastDue60_89"]),
            "NumberOfDependents": int(form_data["Dependents"])
        }

        df = pd.DataFrame([input_data])
        X = preprocessor.transform(df)
        prediction = model.predict(X)[0]

        result = "✅ Khách hàng có khả năng trả nợ tốt" if prediction == 0 else "❌ Khách hàng có nguy cơ vỡ nợ"

    return render_template("index.html", result=result, form_data=form_data)
if __name__ == "__main__":
    app.run(debug=True)