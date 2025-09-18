# Data Analysis @ Horeb Energy Formula UFMG

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com)

A comprehensive web-based data analysis platform for Formula UFMG data, featuring advanced visualization, statistical analysis, and performance optimization tools.

## Features

### **Data Processing & Management**
- **Raw File Upload**: Process CSV files directly from FTManager
- **Automatic Data Cleaning**: Handle missing values, outliers, and data validation
- **Multi-file Analysis**: Compare data across multiple sessions and drivers
- **Real-time Processing**: Fast data processing with progress indicators

### **Advanced Visualization**
- **Interactive Plots**: Line plots, scatter plots, histograms, box plots
- **Correlation Analysis**: Heatmaps and correlation matrices
- **Track Mapping**: Parameter visualization on track layouts
- **Performance Optimization**: Automatic data sampling for large datasets

### **Statistical Analysis**
- **Descriptive Statistics**: Mean, median, standard deviation, quartiles
- **Correlation Detection**: Identify relationships between parameters
- **Driver Behavior Analysis**: Analyze driving patterns and performance
- **Lap Detection**: Automatic lap segmentation and analysis

### **Security & Access Control**
- **Password Protection**: Secure access with session management
- **Data Privacy**: Local processing with no external data transmission
- **Session Management**: Automatic logout and secure authentication

### **Export & Reporting**
- **CSV Export**: Download processed data in standard format
- **Analysis Reports**: Generate comprehensive performance reports
- **Visualization Export**: Save plots and charts for presentations

## Quick Start

### Prerequisites
- Python 3.8 or higher
- Modern web browser (Chrome, Firefox, Safari, Edge)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/KPI_v2.git
cd KPI_v2
```

2. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

3. **Ensure backend modules are available:**
   The following modules should be present in the `backend/` directory:
   - `math_channel.py` - Core mathematical processing
   - `data_configuration.py` - Data configuration and validation
   - `sync.py` - Data synchronization utilities
   - `analysis.py` - Statistical analysis functions
   - `report.py` - Report generation
   - `constants.py` - Application constants

### Running the Application

1. **Start the Flask server:**
```bash
python flask_app.py
```

2. **Access the application:**
   - Open web browser
   - Navigate to: `http://localhost:5000`
   - Enter the password

3. **First-time setup:**
   - Upload a raw CSV file from FTManager
   - Process the file to create analysis-ready data
   - Start exploring data!

## User Guide

### **Workflow Overview**
1. **Upload** → 2. **Process** → 3. **Load** → 4. **Analyze** → 5. **Export**

### **Step 1: Upload and Process Files**
- Navigate to the sidebar "Process Raw File" section
- Click "Upload Raw CSV File" and select FTManager export
- Click "Process File" to clean and process the data
- Monitor progress with the real-time progress bar
- Processed files are automatically saved with timestamps

### **Step 2: Load Data for Analysis**
- Go to "Load Processed Files" section
- Select one or more processed files from the dropdown
- Click "Load Data" to combine datasets
- View comprehensive data information including:
  - Total records and columns
  - Numeric parameters available
  - Time range and duration
  - File details and metadata

### **Step 3: Create Visualizations**
- Use "Analysis Controls" to configure plots:
  - **Plot Type**: Line Plot, Scatter Plot, Histogram, Box Plot, Correlation Matrix
  - **X-Axis**: Time, distance, or any numeric parameter
  - **Y-Axis**: Select multiple parameters for comparison
  - **Color Coding**: Optional parameter for enhanced visualization
- Click "Update Plot" to generate interactive charts
- Large datasets are automatically sampled for optimal performance

### **Step 4: Statistical Analysis**
- Navigate to the "Statistics" tab
- Select specific laps or analyze all data
- Choose parameters for detailed analysis
- Click "Generate Statistics" to get:
  - Descriptive statistics (mean, median, std, quartiles)
  - Correlation matrices with heatmaps
  - Strong correlation identification (|r| > 0.7)

### **Step 5: Export Results**
- **CSV Export**: Download processed data in standard format
- **Report Export**: Generate comprehensive analysis reports
- **Visualization Export**: Save plots for presentations
- All exports include timestamps and metadata

## Project Structure

```
KPI_v2/
├── 📄 flask_app.py              # Flask backend server & API endpoints
├── 📁 templates/
│   ├── index.html               # Main application interface
│   └── login.html               # Authentication page
├── 📁 static/
│   ├── style.css                # Custom CSS styling & dark theme
│   ├── app.js                   # Frontend JavaScript logic
│   └── logo.png                 # Horeb Energy Formula UFMG logo
├── 📁 backend/                  # Core Python processing modules
│   ├── math_channel.py          # Mathematical data processing
│   ├── data_configuration.py    # Data validation & configuration
│   ├── sync.py                  # Data synchronization utilities
│   ├── analysis.py              # Statistical analysis functions
│   ├── report.py                # Report generation
│   └── constants.py             # Application constants
├── 📁 processed_files/          # Auto-generated processed data storage
├── 📄 requirements.txt          # Python dependencies
├── 📄 README.md                 # Project documentation
└── 📄 .gitignore                # Git ignore rules
```

## API Reference

### Authentication Endpoints
- `GET /login` - Login page
- `POST /login` - Authenticate user
- `GET /logout` - Logout and clear session

### Data Management Endpoints
- `GET /api/files` - Get list of available processed files
- `POST /api/upload` - Upload and process raw CSV file
- `POST /api/load` - Load selected processed files

### Analysis Endpoints
- `POST /api/plot` - Create interactive plot based on parameters
- `POST /api/statistics` - Generate statistical analysis
- `POST /api/detect_laps` - Automatic lap detection
- `POST /api/track_analysis` - Track-specific analysis
- `POST /api/driver_behavior` - Driver behavior analysis
- `POST /api/speed_zones` - Speed zone analysis

### Export Endpoints
- `GET /api/export/csv` - Export processed data as CSV
- `GET /api/export/report` - Export comprehensive analysis report

## Deployment Options

### Local Development
```bash
python flask_app.py
# Access at http://localhost:5000
```

### GitHub Pages Deployment

**Note**: GitHub Pages serves static files only and cannot run Flask applications directly. For a live demo, you'll need to deploy to a platform that supports Python/Flask.

#### Option 1: Deploy to Heroku (Recommended)
1. **Create Heroku account** and install Heroku CLI
2. **Create Procfile:**
```bash
echo "web: python flask_app.py" > Procfile
```
3. **Deploy:**
```bash
heroku create your-app-name
git push heroku main
```

#### Option 2: Deploy to Railway
1. **Connect GitHub repository** to Railway
2. **Set environment variables** in Railway dashboard
3. **Deploy automatically** on every push

#### Option 3: Local Development Only
For local development and testing:
```bash
python flask_app.py
# Access at http://localhost:5000
```

### Production Deployment
For production use, consider:
- **WSGI Server**: Use Gunicorn or uWSGI instead of Flask dev server
- **Reverse Proxy**: Nginx or Apache for better performance
- **Environment Variables**: Set secure passwords and secrets
- **HTTPS**: Enable SSL certificates for secure data transmission

## Technology Stack

### Backend
- **Python 3.8+**: Core programming language
- **Flask 2.0+**: Web framework and API server
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Plotly**: Interactive visualization generation
- **Flask-CORS**: Cross-origin resource sharing

### Frontend
- **HTML5**: Semantic markup and structure
- **CSS3**: Custom styling with dark theme
- **JavaScript ES6+**: Modern frontend logic
- **Bootstrap 5**: Responsive UI components
- **Plotly.js**: Interactive chart rendering
- **Font Awesome**: Icon library

### Security Features
- **Session Management**: Secure authentication
- **Password Protection**: Configurable access control
- **Security Headers**: XSS protection, content type options
- **Local Processing**: No external data transmission

## 🔧 Configuration

### Environment Variables
```bash
# Optional: Set custom password
export WEBSITE_PASSWORD="secure_password"

# Optional: Set Flask secret key
export FLASK_SECRET_KEY="secret_key"
```

### Customization
- **Logo**: Replace `static/logo.png` with team logo
- **Colors**: Modify CSS variables in `static/style.css`
- **Password**: Change `WEBSITE_PASSWORD` in `flask_app.py`

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -m "Add feature"`
4. Push to branch: `git push origin feature-name`
5. Submit a Pull Request

## About Horeb Energy Formula UFMG

This tool was developed for the Horeb Energy Formula UFMG team to analyze data from the Formula Student vehicle. It provides comprehensive data analysis capabilities to optimize vehicle performance and driver behavior.

## Support

For questions or support:
- Create an issue in the GitHub repository
- Contact the development team
- Check the documentation for troubleshooting

---

**Made with ❤️ for Formula UFMG**
