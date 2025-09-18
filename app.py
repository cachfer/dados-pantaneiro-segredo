from flask import Flask, render_template, request, jsonify, send_file, session, redirect, url_for
from flask_cors import CORS
from functools import wraps
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
import seaborn as sns
import os
import sys
import io
import json
import base64
from datetime import datetime
import tempfile
import shutil

# Add backend modules to path
backend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend')
sys.path.append(backend_path)

try:
    from backend.math_channel import process_data, code_variables
    BACKEND_AVAILABLE = True
    print("✅ Backend modules loaded successfully")
except ImportError as e:
    BACKEND_AVAILABLE = False
    print(f"❌ Backend modules not available: {e}")
    print("File processing will be limited.")

app = Flask(__name__)
app.secret_key = 'pantaneiro'  # Change this in production
CORS(app)

# Password protection
WEBSITE_PASSWORD = "pantaneiro"  # Change this to your desired password

# Security and performance headers
@app.after_request
def after_request(response):
    # Security headers
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    
    # Remove server identification
    response.headers.pop('Server', None)
    response.headers.pop('X-Powered-By', None)
    
    # Cache control for static resources
    if request.endpoint == 'static':
        response.headers['Cache-Control'] = 'public, max-age=31536000, immutable'
    else:
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    
    # Content type with charset
    if response.content_type and 'text/html' in response.content_type:
        response.headers['Content-Type'] = 'text/html; charset=utf-8'
    elif response.content_type and 'application/json' in response.content_type:
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
    
    return response

# Password protection decorator
def require_password(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('authenticated'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Global variables to store processed data
processed_data = None
processed_files_dir = os.path.join(os.path.dirname(__file__), "processed_files")
os.makedirs(processed_files_dir, exist_ok=True)

def get_available_files():
    """Get list of available processed files"""
    try:
        files = []
        print(f"Looking for files in: {processed_files_dir}")
        print(f"Directory exists: {os.path.exists(processed_files_dir)}")
        if os.path.exists(processed_files_dir):
            all_files = os.listdir(processed_files_dir)
            print(f"All files in directory: {all_files}")
            for file in all_files:
                if file.endswith('.csv') and 'processed' in file:
                    files.append(file)
            print(f"Processed files found: {files}")
        else:
            print(f"Directory does not exist: {processed_files_dir}")
            # Try to create it
            os.makedirs(processed_files_dir, exist_ok=True)
            print(f"Created directory: {processed_files_dir}")
        return sorted(files, reverse=True)
    except Exception as e:
        print(f"Error getting files: {e}")
        import traceback
        traceback.print_exc()
        return []

def load_data(filepath):
    """Load data from CSV file"""
    try:
        if filepath and os.path.exists(filepath):
            return pd.read_csv(filepath)
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def get_numeric_columns(df):
    """Get numeric columns from dataframe"""
    if df is None:
        return []
    return df.select_dtypes(include=[np.number]).columns.tolist()

def get_telemetry_columns(df):
    """Get list of meaningful telemetry columns (excluding calculated metadata)"""
    if df is None:
        return []
    
    # Exclude system columns and metadata
    exclude_patterns = [
        'file_source', 'file_name', 'lap', 'segment', 'index', 'unnamed',
        'sync', 'ext', 'latitude', 'longitude'
    ]
    
    meaningful_cols = []
    for col in df.columns:
        col_lower = col.lower()
        if (df[col].dtype in ['float64', 'int64'] and 
            not any(pattern in col_lower for pattern in exclude_patterns)):
            meaningful_cols.append(col)
    
    return meaningful_cols

def create_plot_data(df, plot_type, x_axis, y_axis, color_axis=None):
    """Create plot data for frontend"""
    if df is None or not y_axis:
        return None
    
    try:
        # For large datasets, sample the data to improve performance
        if len(df) > 2000:
            print(f"Large dataset detected ({len(df)} points). Sampling for better performance...")
            # Sample every nth point to get ~2000 points for better performance
            sample_rate = max(1, len(df) // 2000)
            df = df.iloc[::sample_rate].copy()
            print(f"Sampled to {len(df)} points (every {sample_rate}th point)")
        
        print(f"Creating {plot_type} plot with {len(df)} data points")
        if plot_type == "Line Plot":
            fig = go.Figure()
            colors_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            
            y_params = y_axis if isinstance(y_axis, list) else [y_axis]
            
            for i, y_param in enumerate(y_params):
                if y_param in df.columns:
                    print(f"Adding trace for {y_param}, data range: {df[y_param].min():.2f} to {df[y_param].max():.2f}")
                    
                    # Use connectgaps=False and reduce marker size for better performance
                    fig.add_trace(go.Scatter(
                        x=df[x_axis] if x_axis else range(len(df)),
                        y=df[y_param],
                        mode='lines',
                        name=y_param,
                        line=dict(color=colors_list[i % len(colors_list)], width=1),
                        connectgaps=False,
                        hovertemplate=f'<b>{y_param}</b><br>' +
                                    f'{x_axis or "Index"}: %{{x}}<br>' +
                                    f'{y_param}: %{{y}}<br>' +
                                    '<extra></extra>'
                    ))
                else:
                    print(f"Warning: Column {y_param} not found in data")
            
            fig.update_layout(
                title=f"{', '.join(y_params)} vs {x_axis or 'Index'}",
                xaxis_title=x_axis or "Index",
                yaxis_title=y_params[0] if len(y_params) == 1 else "Value",
                height=500,
                template='plotly_dark',
                # Optimize for performance
                showlegend=True,
                hovermode='closest',
                # Reduce animation and transitions
                transition={'duration': 0}
            )
            
        elif plot_type == "Scatter Plot":
            fig = px.scatter(
                df, 
                x=x_axis, 
                y=y_axis[0] if isinstance(y_axis, list) else y_axis,
                color=color_axis if color_axis != "None" else None,
                title=f"{y_axis[0] if isinstance(y_axis, list) else y_axis} vs {x_axis}",
                height=500
            )
            fig.update_layout(template='plotly_dark')
            
        elif plot_type == "Histogram":
            fig = go.Figure()
            colors_list = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7']
            
            y_params = y_axis if isinstance(y_axis, list) else [y_axis]
            
            for i, y_param in enumerate(y_params):
                if y_param in df.columns:
                    fig.add_trace(go.Histogram(
                        x=df[y_param],
                        name=y_param,
                        marker_color=colors_list[i % len(colors_list)],
                        opacity=0.7
                    ))
            
            fig.update_layout(
                title=f"Distribution of {', '.join(y_params)}",
                xaxis_title="Value",
                yaxis_title="Frequency",
                height=500,
                barmode='overlay',
                template='plotly_dark'
            )
            
        elif plot_type == "Box Plot":
            fig = go.Figure()
            colors_list = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7']
            
            y_params = y_axis if isinstance(y_axis, list) else [y_axis]
            
            for i, y_param in enumerate(y_params):
                if y_param in df.columns:
                    fig.add_trace(go.Box(
                        y=df[y_param],
                        name=y_param,
                        marker_color=colors_list[i % len(colors_list)]
                    ))
            
            fig.update_layout(
                title=f"Box Plot of {', '.join(y_params)}",
                yaxis_title="Value",
                height=500,
                template='plotly_dark'
            )
            
        elif plot_type == "Correlation Matrix":
            numeric_cols = get_numeric_columns(df)
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                fig = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale="RdBu_r",
                    title="Correlation Matrix"
                )
                fig.update_layout(height=600, template='plotly_dark')
            else:
                return None
        else:
            return None
            
        plot_json = fig.to_json()
        print(f"Plot created successfully, JSON length: {len(plot_json)}")
        return plot_json
        
    except Exception as e:
        print(f"Error creating plot: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

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
    files = get_available_files()
    return jsonify({'files': files})

@app.route('/api/upload', methods=['POST'])
@require_password
def upload_file():
    """Handle file upload and processing"""
    print("Upload request received")
    
    if not BACKEND_AVAILABLE:
        print("Backend not available")
        return jsonify({'error': 'Backend modules not available'}), 500
    
    if 'file' not in request.files:
        print("No file in request")
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    print(f"File received: {file.filename}, size: {file.content_length}")
    
    if file.filename == '':
        print("Empty filename")
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.endswith('.csv'):
        print(f"Invalid file type: {file.filename}")
        return jsonify({'error': 'Only CSV files are supported'}), 400
    
    try:
        # Save uploaded file temporarily
        temp_path = os.path.join(processed_files_dir, f"temp_{file.filename}")
        print(f"Saving file to: {temp_path}")
        file.save(temp_path)
        
        # Process the file
        print("Starting file processing...")
        df_cleaned = code_variables(temp_path)
        print("File cleaned, starting data processing...")
        df_processed = process_data(df_cleaned)
        print("Data processing completed")
        
        # Save processed file
        base_name = os.path.splitext(file.filename)[0]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_name = f"{base_name}_processed_{timestamp}.csv"
        output_path = os.path.join(processed_files_dir, output_name)
        
        df_processed.to_csv(output_path, index=False)
        print(f"Processed file saved to: {output_path}")
        print(f"File exists: {os.path.exists(output_path)}")
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return jsonify({
            'success': True,
            'message': f'File processed successfully! Saved as: {output_name}',
            'filename': output_name
        })
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        
        # Clean up temp file on error
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

@app.route('/api/load', methods=['POST'])
@require_password
def load_files():
    """Load selected processed files"""
    data = request.get_json()
    selected_files = data.get('files', [])
    
    if not selected_files:
        return jsonify({'error': 'No files selected'}), 400
    
    try:
        combined_data = None
        file_info = []
        
        for i, selected_file in enumerate(selected_files):
            filepath = os.path.join(processed_files_dir, selected_file)
            data = load_data(filepath)
            if data is not None:
                data['file_source'] = f"File_{i+1}_{selected_file[:20]}"
                data['file_name'] = selected_file
                
                if combined_data is None:
                    combined_data = data
                else:
                    combined_data = pd.concat([combined_data, data], ignore_index=True)
                
                file_info.append({
                    'File': f"File_{i+1}",
                    'Name': selected_file,
                    'Records': len(data)
                })
        
        if combined_data is not None:
            global processed_data
            processed_data = combined_data
            
            # Get column information
            numeric_cols = get_numeric_columns(combined_data)
            all_cols = combined_data.columns.tolist()
            
            return jsonify({
                'success': True,
                'message': f'Loaded {len(selected_files)} file(s) for comparison',
                'data_info': {
                    'total_records': len(combined_data),
                    'total_columns': len(all_cols),
                    'numeric_columns': len(numeric_cols),
                    'columns': all_cols,
                    'numeric_columns_list': numeric_cols
                },
                'file_info': file_info
            })
        else:
            return jsonify({'error': 'Failed to load data'}), 500
            
    except Exception as e:
        return jsonify({'error': f'Error loading files: {str(e)}'}), 500

@app.route('/api/data/info', methods=['GET'])
def get_data_info():
    """Get information about currently loaded data"""
    global processed_data
    
    if processed_data is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    numeric_cols = get_numeric_columns(processed_data)
    all_cols = processed_data.columns.tolist()
    
    return jsonify({
        'total_records': len(processed_data),
        'total_columns': len(all_cols),
        'numeric_columns': len(numeric_cols),
        'columns': all_cols,
        'numeric_columns_list': numeric_cols,
        'time_range': {
            'min': float(processed_data['time'].min()) if 'time' in processed_data.columns else None,
            'max': float(processed_data['time'].max()) if 'time' in processed_data.columns else None
        }
    })

@app.route('/api/plot', methods=['POST'])
@require_password
def create_plot():
    """Create a plot based on parameters"""
    global processed_data
    
    if processed_data is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    data = request.get_json()
    plot_type = data.get('type', 'Line Plot')
    x_axis = data.get('x_axis')
    y_axis = data.get('y_axis')
    color_axis = data.get('color_axis')
    
    print(f"Plot request: type={plot_type}, x_axis={x_axis}, y_axis={y_axis}")
    print(f"Data shape: {processed_data.shape}, columns: {list(processed_data.columns)[:10]}...")
    
    plot_json = create_plot_data(processed_data, plot_type, x_axis, y_axis, color_axis)
    
    if plot_json:
        return jsonify({'success': True, 'plot': plot_json})
    else:
        return jsonify({'error': 'Failed to create plot'}), 500

@app.route('/api/data/preview', methods=['GET'])
def get_data_preview():
    """Get data preview for overview tab"""
    global processed_data
    
    if processed_data is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    # Get query parameters
    limit = request.args.get('limit', 100, type=int)
    search_column = request.args.get('search_column', 'All')
    search_value = request.args.get('search_value', '')
    
    # Filter data if search is specified
    display_df = processed_data.head(limit)
    
    if search_column != "All" and search_value:
        mask = processed_data[search_column].astype(str).str.contains(search_value, case=False, na=False)
        display_df = processed_data[mask].head(limit)
    
    # Convert to JSON-serializable format
    data_dict = display_df.to_dict('records')
    
    return jsonify({
        'success': True,
        'data': data_dict,
        'total_records': len(processed_data),
        'displayed_records': len(display_df)
    })

@app.route('/api/statistics', methods=['POST'])
def get_statistics():
    """Get statistical analysis of data"""
    global processed_data
    
    if processed_data is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    data = request.get_json()
    selected_laps = data.get('selected_laps', [])
    
    # Filter data based on lap selection
    if selected_laps and 'lap' in processed_data.columns:
        stats_df = processed_data[processed_data['lap'].isin(selected_laps)].copy()
    else:
        stats_df = processed_data.copy()
    
    numeric_df = stats_df.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) == 0:
        return jsonify({'error': 'No numeric columns found'}), 400
    
    # Descriptive statistics
    desc_stats = numeric_df.describe().to_dict()
    
    # Correlation analysis
    meaningful_cols = get_telemetry_columns(stats_df)
    correlation_data = None
    high_corr_pairs = []
    
    if len(meaningful_cols) > 1:
        corr_df = stats_df[meaningful_cols]
        corr_matrix = corr_df.corr()
        correlation_data = corr_matrix.to_dict()
        
        # Find strongest correlations
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        for col in upper_tri.columns:
            for idx in upper_tri.index:
                if pd.notna(upper_tri.loc[idx, col]) and abs(upper_tri.loc[idx, col]) > 0.7:
                    high_corr_pairs.append({
                        'Variable 1': idx,
                        'Variable 2': col,
                        'Correlation': f"{upper_tri.loc[idx, col]:.3f}",
                        'Strength': 'Very Strong' if abs(upper_tri.loc[idx, col]) > 0.9 else 'Strong'
                    })
    
    return jsonify({
        'success': True,
        'descriptive_stats': desc_stats,
        'correlation_matrix': correlation_data,
        'high_correlations': high_corr_pairs,
        'meaningful_columns': meaningful_cols
    })

@app.route('/api/export/csv', methods=['GET'])
def export_csv():
    """Export data as CSV"""
    global processed_data
    
    if processed_data is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    processed_data.to_csv(temp_file.name, index=False)
    temp_file.close()
    
    return send_file(
        temp_file.name,
        as_attachment=True,
        download_name=f"formula_ufmg_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mimetype='text/csv'
    )

@app.route('/api/export/report', methods=['GET'])
def export_report():
    """Export analysis report"""
    global processed_data
    
    if processed_data is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    # Generate report
    report = f"""
FORMULA UFMG - DATA ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
=====================================

DATASET OVERVIEW:
- Total Records: {len(processed_data):,}
- Total Columns: {len(processed_data.columns)}
- Numeric Columns: {len(get_numeric_columns(processed_data))}
- Time Range: {processed_data['time'].min():.1f}s - {processed_data['time'].max():.1f}s (Duration: {(processed_data['time'].max() - processed_data['time'].min()):.1f}s)

KEY PERFORMANCE METRICS:
"""
    
    # Add performance metrics if available
    if 'traction_speed' in processed_data.columns:
        report += f"- Max Speed: {processed_data['traction_speed'].max():.1f} km/h\n"
        report += f"- Avg Speed: {processed_data['traction_speed'].mean():.1f} km/h\n"
    
    if 'G_combined' in processed_data.columns:
        report += f"- Max G-Force: {processed_data['G_combined'].max():.2f} G\n"
        report += f"- Avg G-Force: {processed_data['G_combined'].mean():.2f} G\n"
    
    if 'rpm' in processed_data.columns:
        report += f"- Max RPM: {processed_data['rpm'].max():.0f}\n"
        report += f"- Avg RPM: {processed_data['rpm'].mean():.0f}\n"
    
    # Add column summary
    report += f"\nCOLUMN SUMMARY:\n"
    for col in processed_data.columns:
        if processed_data[col].dtype in ['float64', 'int64']:
            report += f"- {col}: {processed_data[col].mean():.3f} ± {processed_data[col].std():.3f} (range: {processed_data[col].min():.3f} to {processed_data[col].max():.3f})\n"
    
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.txt', mode='w')
    temp_file.write(report)
    temp_file.close()
    
    return send_file(
        temp_file.name,
        as_attachment=True,
        download_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mimetype='text/plain'
    )

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

# Vercel compatibility
app = app
