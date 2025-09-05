from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
from werkzeug.utils import secure_filename
import os
import logging
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load trained model and pipeline
try:
    model = joblib.load('../models/best_model.pkl')
    feature_pipeline = joblib.load('../models/feature_pipeline.pkl')
    logging.info("Model and pipeline loaded successfully")
except Exception as e:
    logging.error(f"Error loading model: {str(e)}")
    model = None
    feature_pipeline = None

@app.route('/')
def home():
    """Render home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Make prediction from form data"""
    if model is None or feature_pipeline is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get form data
        form_data = request.form.to_dict()
        
        # Convert to DataFrame
        input_data = pd.DataFrame([form_data])
        
        # Convert numeric fields
        numeric_fields = ['RhythmScore', 'AudioLoudness', 'VocalContent', 
                         'AcousticQuality', 'InstrumentalScore', 'LivePerformanceLikelihood',
                         'MoodScore', 'TrackDurationMs', 'Energy']
        
        for field in numeric_fields:
            if field in input_data.columns:
                input_data[field] = pd.to_numeric(input_data[field], errors='coerce')
        
        # Handle missing values
        input_data = input_data.fillna(0)
        
        # Apply feature engineering
        processed_data = feature_pipeline.transform(input_data)
        
        # Make prediction
        prediction = model.predict(processed_data)[0]
        
        return jsonify({
            'prediction': round(prediction, 2),
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Handle batch prediction from CSV file"""
    if model is None or feature_pipeline is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and file.filename.endswith('.csv'):
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Load and process data
            data = pd.read_csv(filepath)
            
            # Store original data for response
            original_data = data.copy()
            
            # Apply feature engineering
            processed_data = feature_pipeline.transform(data)
            
            # Make predictions
            predictions = model.predict(processed_data)
            
            # Add predictions to original data
            original_data['Predicted_BPM'] = predictions.round(2)
            
            # Create response
            response_data = original_data.to_dict('records')
            
            # Generate output file
            output_filename = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
            original_data.to_csv(output_path, index=False)
            
            return jsonify({
                'predictions': response_data,
                'download_link': f'/download/{output_filename}',
                'status': 'success'
            })
        
        else:
            return jsonify({'error': 'Invalid file format. Please upload a CSV file'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/download/<filename>')
def download_file(filename):
    """Download prediction results"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

@app.route('/model_info')
def model_info():
    """Get model information"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    model_info = {
        'model_type': type(model).__name__,
        'features_used': feature_pipeline.named_steps['feature_engineering'].feature_names if hasattr(feature_pipeline, 'named_steps') else 'Unknown',
        'training_date': '2024-01-01'  # You can store this during training
    }
    
    return jsonify(model_info)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)