from flask import Flask, render_template, request, send_file
import pandas as pd
import pickle
import plotly.express as px
import plotly
import json
import os

app = Flask(__name__)

# Load model and preprocessor
with open('model/tourism_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('model/preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

# Load dataset
df = pd.read_csv('data/combined_tourism_dataset_labeled.csv')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    input_data = {
        'Place_Type': request.form['place_type'],
        'Tourist_Type': request.form['tourist_type'],
        'Crime_Level': request.form['crime_level'],
        'Healthcare_Quality': request.form['healthcare_quality'],
        'Political_Stability': request.form['political_stability'],
        'Emergency_Services': request.form['emergency_services'],
        'Seasonal_Safety_Factors': request.form['seasonal_safety'],
        'Recent_Incident_Level': request.form['recent_incident'],
        'Tourist_Reviews_Safety': request.form['tourist_reviews'],
        'Transportation_Access': request.form['transport_access'],
        'Cultural_Friendliness': request.form['cultural_friendliness'],
        'Environmental_Risks': request.form['env_risks'],
        'Language_Accessibility': request.form['lang_access'],
        'Tourist_Facilities': request.form['tourist_facilities'],
        'Safety_Education_Awareness': request.form['safety_education'],
        'Local_Law_Enforcement': request.form['law_enforcement'],
        'Accessibility_for_Disabled': request.form['access_disabled'],
        'Night_Safety': request.form['night_safety']
    }
    input_df = pd.DataFrame([input_data])
    
    # Preprocess and predict
    input_processed = preprocessor.transform(input_df)
    prediction = model.predict(input_processed)[0]
    
    # Save prediction to CSV
    output_df = input_df.copy()
    output_df['Predicted_Safety_Score'] = prediction
    output_df.to_csv('static/prediction.csv', index=False)
    
    return render_template('results.html', prediction=prediction)

@app.route('/dashboard')
def dashboard():
    # Plot average safety score by Place_Type
    avg_scores = df.groupby('Place_Type')['score'].mean().reset_index()
    fig = px.bar(avg_scores, x='Place_Type', y='score', title='Average Safety Score by Place Type')
    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('dashboard.html', graph_json=graph_json)

@app.route('/download')
def download():
    return send_file('static/prediction.csv', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)