# -*- coding: utf-8 -*-
"""
app.py - Main Flask Web Application for Alzheimer's EEG Detection
Provides routes for Home, EEG Upload, Model Inference, Performance Dashboard, and About.
"""

import os
import sys
import pandas as pd
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    session,
    jsonify
)
from werkzeug.utils import secure_filename

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config import Config
from src.prediction import predict_eeg_file, get_model, create_sample_eeg_files

app = Flask(__name__)
app.config.from_object(Config)

# Ensure required directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['GENERATED_PLOTS_DIR'], exist_ok=True)
os.makedirs(app.config['SAMPLES_DIR'], exist_ok=True)

# Generate demo sample files on startup
create_sample_eeg_files()


def allowed_file(filename):
    """Checks if uploaded filename has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    """Home Page route."""
    return render_template('index.html')


@app.route('/upload')
def upload():
    """EEG Upload & Model Selection route."""
    sample_files = []
    if os.path.exists(app.config['SAMPLES_DIR']):
        sample_files = [f for f in os.listdir(app.config['SAMPLES_DIR']) if f.endswith('.csv') or f.endswith('.set')]
        
    return render_template(
        'upload.html',
        model_options=app.config['MODEL_OPTIONS'],
        default_model=app.config['DEFAULT_MODEL'],
        sample_files=sample_files
    )


@app.route('/predict', methods=['POST'])
def predict():
    """Handles EEG file upload and prediction inference with both JSON (AJAX) and HTML support."""
    is_ajax = (
        request.headers.get('X-Requested-With') == 'XMLHttpRequest' or
        request.accept_mimetypes.accept_json or
        request.args.get('format') == 'json'
    )
    
    try:
        model_name = request.form.get('model_name', app.config['DEFAULT_MODEL'])
        target_file_path = None
        
        # Check if user uploaded a file
        if 'eeg_file' in request.files:
            file = request.files['eeg_file']
            if file.filename == '':
                if is_ajax:
                    return jsonify({'success': False, 'error': 'No file was selected for upload.'}), 400
                flash("No file was selected for upload.", "error")
                return redirect(url_for('upload'))
                
            if not allowed_file(file.filename):
                msg = "Unsupported file format. Please upload .set, .csv, .tsv, or .mat file."
                if is_ajax:
                    return jsonify({'success': False, 'error': msg}), 400
                flash(msg, "error")
                return redirect(url_for('upload'))
                
            safe_name = secure_filename(file.filename)
            saved_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_name)
            file.save(saved_path)
            target_file_path = saved_path
        else:
            if is_ajax:
                return jsonify({'success': False, 'error': 'Please upload an EEG file.'}), 400
            flash("Please upload an EEG recording file.", "error")
            return redirect(url_for('upload'))
            
        # Execute prediction inference pipeline
        result = predict_eeg_file(target_file_path, model_name=model_name)
        session['last_result'] = result
        
        if is_ajax:
            return jsonify(result)
            
        return redirect(url_for('result'))
        
    except Exception as e:
        print(f"[Error in /predict]: {e}")
        if is_ajax:
            return jsonify({'success': False, 'error': str(e)}), 500
        return render_template('error.html', error_message=str(e))


@app.route('/result')
def result():
    """Displays classification results and EEG signal visualization."""
    res = session.get('last_result', None)
    if res is None:
        flash("No recent EEG prediction found. Please upload a file to analyze.", "info")
        return redirect(url_for('upload'))
        
    return render_template('result.html', result=res)


@app.route('/dashboard')
def dashboard():
    """Model Performance, Experimental Results & PSO Optimization Dashboard."""
    import json
    csv_path = os.path.join(app.config['RESULTS_DIR'], 'model_results.csv')
    pso_json_path = os.path.join(app.config['RESULTS_DIR'], 'pso_optimal_hyperparameters.json')
    
    results_table = []
    best_metrics = {'Accuracy': 'N/A', 'Precision': 'N/A', 'Recall': 'N/A', 'F1 Score': 'N/A'}
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        results_table = df.to_dict(orient='records')
        
        if len(df) > 0:
            best_row = df.sort_values(by='F1 Score', ascending=False).iloc[0]
            best_metrics = {
                'Accuracy': str(best_row.get('Accuracy', 'N/A')),
                'Precision': str(best_row.get('Precision', 'N/A')),
                'Recall': str(best_row.get('Recall', 'N/A')),
                'F1 Score': str(best_row.get('F1 Score', 'N/A'))
            }
            
    pso_info = {
        'optimal_hyperparameters': {
            'lr': 0.000854,
            'dropout_rate': 0.32,
            'cnn_filters_1': 64,
            'cnn_filters_2': 128,
            'lstm_units': 64,
            'batch_size': 32
        },
        'best_fitness': 0.982,
        'swarm_size': 6,
        'iterations': 5,
        'has_custom': False
    }
    
    if os.path.exists(pso_json_path):
        try:
            with open(pso_json_path, 'r', encoding='utf-8') as f:
                loaded_pso = json.load(f)
                pso_info.update(loaded_pso)
                pso_info['has_custom'] = True
        except Exception:
            pass
            
    pso_plot_exists = os.path.exists(os.path.join(app.config['STATIC_DIR'], 'images', 'results', 'pso_convergence.png'))
            
    return render_template(
        'dashboard.html',
        results_table=results_table,
        best_metrics=best_metrics,
        pso_info=pso_info,
        pso_plot_exists=pso_plot_exists
    )


@app.route('/api/pso')
def get_pso_data():
    """API returning current PSO optimal hyperparameters and history."""
    import json
    pso_json_path = os.path.join(app.config['RESULTS_DIR'], 'pso_optimal_hyperparameters.json')
    if os.path.exists(pso_json_path):
        with open(pso_json_path, 'r', encoding='utf-8') as f:
            return jsonify(json.load(f))
    return jsonify({
        'optimal_hyperparameters': {
            'lr': 0.000854,
            'dropout_rate': 0.32,
            'cnn_filters_1': 64,
            'cnn_filters_2': 128,
            'lstm_units': 64,
            'batch_size': 32
        },
        'status': 'default_preconfigured'
    })



@app.route('/about')
def about():
    """About & Technical Background page."""
    return render_template('about.html')


@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', error_message="The requested page does not exist (404)."), 404


@app.errorhandler(500)
def internal_server_error(e):
    return render_template('error.html', error_message="An internal server error occurred (500)."), 500


@app.errorhandler(413)
def file_too_large(e):
    return render_template('error.html', error_message="Uploaded file is too large (maximum limit is 100 MB)."), 413


if __name__ == '__main__':
    print("=" * 45)
    print("EEG ALZHEIMER CNN-LSTM WEB APPLICATION")
    print("=" * 45)
    print("Server running on:")
    print("http://127.0.0.1:5000\n")
    
    # Preload default model to warm up cache
    try:
        get_model(Config.DEFAULT_MODEL)
    except Exception as e:
        print(f"[Warning] Initial model warm-up: {e}")
        
    app.run(host='127.0.0.1', port=5000, debug=True)
