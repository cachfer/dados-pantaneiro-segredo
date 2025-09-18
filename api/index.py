from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from functools import wraps
import os
import json
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'pantaneiro'

# Enable CORS manually
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response

# Password protection
WEBSITE_PASSWORD = "pantaneiro"

# Password protection decorator
def require_password(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('authenticated'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if request.method == 'POST':
        password = request.form.get('password')
        if password == WEBSITE_PASSWORD:
            session['authenticated'] = True
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Invalid password')
    return render_template('login.html')

@app.route('/logout')
def logout():
    """Logout and clear session"""
    session.pop('authenticated', None)
    return redirect(url_for('login'))

@app.route('/')
@require_password
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/api/files', methods=['GET'])
@require_password
def get_files():
    """Get list of available processed files"""
    return jsonify({'files': []})

@app.route('/api/upload', methods=['POST'])
@require_password
def upload_file():
    """Upload and process file"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})
    
    try:
        # Simple file processing for Vercel
        content = file.read().decode('utf-8')
        lines = content.split('\n')
        headers = lines[0].split(',') if lines else []
        
        return jsonify({
            'success': True, 
            'message': f'File processed successfully. {len(lines)} rows, {len(headers)} columns.',
            'data_info': {
                'total_records': len(lines),
                'total_columns': len(headers),
                'numeric_columns': len(headers),
                'columns': headers,
                'numeric_columns_list': headers
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': f'Error processing file: {str(e)}'})

@app.route('/api/plot', methods=['POST'])
@require_password
def create_plot():
    """Create plot"""
    try:
        data = request.json
        plot_type = data.get('type', 'Line Plot')
        
        # Create a simple sample plot data
        sample_data = {
            'data': [{
                'x': list(range(100)),
                'y': [i * 0.1 + (i % 10) for i in range(100)],
                'type': 'scatter',
                'mode': 'lines',
                'name': 'Sample Data'
            }],
            'layout': {
                'title': f'{plot_type} - Sample Data',
                'template': 'plotly_dark'
            }
        }
        
        return jsonify({
            'success': True,
            'plot': json.dumps(sample_data)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': f'Error creating plot: {str(e)}'})

# Vercel compatibility
handler = app
