from flask import Flask, render_template, request
import pandas as pd
import joblib  # Assuming you saved your model using joblib

app = Flask(__name__)

# Load your pre-trained model (replace with your model file)
logistic_model = joblib.load('../Final_Data/Resources/log_regression_model.pkl')

@app.route('/')
def home():
    return render_template('test.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract form data
    age = int(request.form['Age'])
    sleep_duration = float(request.form['Sleep_Duration'])
    quality_of_sleep = int(request.form['Quality_of_Sleep'])
    physical_activity_level = int(request.form['Physical_Activity_Level'])
    stress_level = int(request.form['Stress_Level'])
    heart_rate = int(request.form['Heart_Rate'])
    daily_steps = int(request.form['Daily_Steps'])
    systolic = int(request.form['systolic'])
    diastolic = int(request.form['diastolic'])
    
    # Handle Gender
    gender_female = 1 if request.form['Gender'] == 'Female' else 0
    gender_male = 1 if request.form['Gender'] == 'Male' else 0

    # Initialize dummy variables for Occupation
    occupation_columns = ['Occupation_Accountant', 'Occupation_Doctor', 'Occupation_Engineer', 
                          'Occupation_Lawyer', 'Occupation_Manager', 'Occupation_Nurse', 
                          'Occupation_Sales Representative', 'Occupation_Salesperson', 
                          'Occupation_Scientist', 'Occupation_Software Engineer', 
                          'Occupation_Teacher']
    
    # Set all occupation dummies to 0 initially
    occupation_data = {col: 0 for col in occupation_columns}
    # Set the selected occupation to 1
    selected_occupation = request.form['Occupation']
    occupation_data[f'Occupation_{selected_occupation}'] = 1

    # Initialize dummy variables for BMI Category
    bmi_columns = ['BMI Category_Normal', 'BMI Category_Obese', 'BMI Category_Overweight']
    # Set all BMI dummies to 0 initially
    bmi_data = {col: 0 for col in bmi_columns}
    # Set the selected BMI category to 1
    selected_bmi = request.form['BMI_Category']
    bmi_data[f'BMI Category_{selected_bmi}'] = 1

    # Combine all data into a single dictionary
    input_data = {
        'Age': age,
        'Sleep Duration': sleep_duration,
        'Quality of Sleep': quality_of_sleep,
        'Physical Activity Level': physical_activity_level,
        'Stress Level': stress_level,
        'Heart Rate': heart_rate,
        'Daily Steps': daily_steps,
        'systolic': systolic,
        'diastolic': diastolic,
        'Gender_Female': gender_female,
        'Gender_Male': gender_male,
        **occupation_data,  # Unpack occupation dummy variables
        **bmi_data  # Unpack BMI dummy variables
    }

    
    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
   
    print(input_df)
    # Make prediction using the logistic regression model
    prediction = logistic_model.predict(input_df)
    
    # Output prediction result
    return f'Predicted Sleep Disorder: {prediction[0]}'

if __name__ == '__main__':
    app.run(debug=True)