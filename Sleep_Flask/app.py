from flask import Flask, render_template, request
import pandas as pd
import joblib  # Assuming you saved your model using joblib
app = Flask(__name__)
# Load your pre-trained model
logistic_model = joblib.load('../Final_Data/Resources/log_regression_model.pkl')
@app.route('/')
def home():
    return render_template('test.html')
@app.route('/predict', methods=['POST'])
def predict():
    # Extract form data
    input_data = {
        'Age': int(request.form['Age']),
        'Sleep Duration': float(request.form['Sleep_Duration']),
        'Quality of Sleep': int(request.form['Quality_of_Sleep']),
        'Physical Activity Level': int(request.form['Physical_Activity_Level']),
        'Stress Level': int(request.form['Stress_Level']),
        'Heart Rate': int(request.form['Heart_Rate']),
        'Daily Steps': int(request.form['Daily_Steps']),
        'systolic': int(request.form['systolic']),
        'diastolic': int(request.form['diastolic']),
        'Gender_Female': bool(request.form.get('Gender_Female', False))
        # Include all other features based on the columns in your dataset
    }
    # Create a DataFrame from the input data
    input_df = pd.DataFrame([input_data])
    # Make a prediction using the logistic regression model
    prediction = logistic_model.predict(input_df)
    # Output prediction result
    return f'Predicted Sleep Disorder: {prediction[0]}'
if __name__ == '__main__':
    app.run(debug=True)