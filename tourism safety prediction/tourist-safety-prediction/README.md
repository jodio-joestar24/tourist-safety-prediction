# Tourism Safety Prediction

## Overview
A final-year MCA project to predict tourism safety scores using a Random Forest Regressor. Built with Flask, scikit-learn, and Pandas, it includes a web interface for predictions and dataset insights.

## Setup Instructions
1. **Prerequisites**: Python 3.8+ installed.
2. Clone the project:
   ```bash
   git clone <repository_url>
   cd tourism_safety_prediction
   ```
3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Place `combined_tourism_dataset_labeled.csv` in the `data/` folder.
6. Train the model:
   ```bash
   python model/train_model.py
   ```
7. Run the Flask app:
   ```bash
   python app.py
   ```
8. Open `http://127.0.0.1:5000` in your browser.

## Usage
- **Home Page**: Input tourism features (e.g., Place Type, Crime Level) to predict safety scores.
- **Dashboard**: View average safety scores by Place Type.
- **Download**: Get prediction results as a CSV file.

## Notes
- Ensure the CSV file is in `data/`.
- Runs locally; no database required.
- For MCA submission, use `report_template.tex` for documentation.

## System Requirements
- OS: Windows/macOS/Linux
- RAM: 4GB+ recommended
- No GPU needed