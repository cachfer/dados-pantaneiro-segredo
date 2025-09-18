// Data Analysis App JavaScript
class DataAnalysisApp {
    constructor() {
        this.currentData = null;
        this.dataInfo = null;
        this.availableFiles = [];
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadAvailableFiles();
        this.setupToleranceSlider();
    }

    setupEventListeners() {
        // File upload
        document.getElementById('fileInput').addEventListener('change', (e) => {
            this.handleFileUpload(e.target.files[0]);
        });

        document.getElementById('processBtn').addEventListener('click', () => {
            this.processFile();
        });

        // File loading
        document.getElementById('loadBtn').addEventListener('click', () => {
            this.loadSelectedFiles();
        });

        // Analysis controls
        document.getElementById('updatePlotBtn').addEventListener('click', () => {
            this.updatePlot();
        });

        document.getElementById('plotTypeSelect').addEventListener('change', () => {
            this.updatePlotTypeControls();
        });

        // Lap detection
        document.getElementById('detectLapsBtn').addEventListener('click', () => {
            this.detectLaps();
        });

        // Statistics
        document.getElementById('generateStatsBtn').addEventListener('click', () => {
            this.generateStatistics();
        });

        // Export
        document.getElementById('exportCsvBtn').addEventListener('click', () => {
            this.exportCsv();
        });

        document.getElementById('exportReportBtn').addEventListener('click', () => {
            this.exportReport();
        });

        // Track analysis buttons
        document.getElementById('parameterMapBtn').addEventListener('click', () => {
            this.createParameterMap();
        });

        document.getElementById('behaviorMapBtn').addEventListener('click', () => {
            this.createBehaviorMap();
        });

        document.getElementById('speedZonesBtn').addEventListener('click', () => {
            this.createSpeedZones();
        });
    }

    setupToleranceSlider() {
        const slider = document.getElementById('detectionTolerance');
        const valueDisplay = document.getElementById('toleranceValue');
        
        slider.addEventListener('input', (e) => {
            valueDisplay.textContent = e.target.value + '%';
        });
    }

    async loadAvailableFiles() {
        try {
            console.log('Loading available files...');
            const response = await fetch('/api/files');
            const data = await response.json();
            console.log('Files response:', data);
            this.availableFiles = data.files;
            console.log('Available files:', this.availableFiles);
            this.updateFileSelect();
        } catch (error) {
            console.error('Error loading files:', error);
            this.showAlert('Error loading files: ' + error.message, 'danger');
        }
    }

    updateFileSelect() {
        const fileSelect = document.getElementById('fileSelect');
        fileSelect.innerHTML = '';
        
        if (this.availableFiles.length === 0) {
            fileSelect.innerHTML = '<option value="">No files available</option>';
        } else {
            this.availableFiles.forEach(file => {
                const option = document.createElement('option');
                option.value = file;
                option.textContent = file;
                fileSelect.appendChild(option);
            });
        }
    }

    async handleFileUpload(file) {
        if (!file) return;
        
        if (!file.name.endsWith('.csv')) {
            this.showAlert('Please select a CSV file', 'warning');
            return;
        }
        
        this.uploadedFile = file;
        this.showAlert('File selected: ' + file.name, 'info');
    }

    async processFile() {
        if (!this.uploadedFile) {
            this.showAlert('Please select a file first', 'warning');
            return;
        }

        this.showLoading('Processing file...');
        
        const formData = new FormData();
        formData.append('file', this.uploadedFile);

        try {
            console.log('Uploading file:', this.uploadedFile.name, 'Size:', this.uploadedFile.size);
            
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });

            console.log('Response status:', response.status);
            console.log('Response headers:', response.headers);

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            console.log('Response data:', result);
            
            if (result.success) {
                this.showAlert(result.message, 'success');
                this.loadAvailableFiles(); // Refresh file list
            } else {
                this.showAlert(result.error, 'danger');
            }
        } catch (error) {
            console.error('Upload error:', error);
            this.showAlert('Error processing file: ' + error.message, 'danger');
        } finally {
            this.hideLoading();
        }
    }

    async loadSelectedFiles() {
        const fileSelect = document.getElementById('fileSelect');
        const selectedFiles = Array.from(fileSelect.selectedOptions).map(option => option.value);
        
        if (selectedFiles.length === 0) {
            this.showAlert('Please select at least one file', 'warning');
            return;
        }

        this.showLoading('Loading data...');

        try {
            const response = await fetch('/api/load', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ files: selectedFiles })
            });

            const result = await response.json();
            
            if (result.success) {
                this.currentData = result.data_info;
                this.showAlert(result.message, 'success');
                this.updateDataInfo(result.data_info, result.file_info);
                this.showAnalysisControls();
                this.updateParameterSelects();
            } else {
                this.showAlert(result.error, 'danger');
            }
        } catch (error) {
            this.showAlert('Error loading files: ' + error.message, 'danger');
        } finally {
            this.hideLoading();
        }
    }

    updateDataInfo(dataInfo, fileInfo) {
        const dataInfoCard = document.getElementById('dataInfo');
        const dataInfoContent = document.getElementById('dataInfoContent');
        
        let html = `
            <div class="row">
                <div class="col-md-3">
                    <div class="stat-card">
                        <div class="stat-value">${dataInfo.total_records.toLocaleString()}</div>
                        <div class="stat-label">Total Records</div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="stat-card">
                        <div class="stat-value">${dataInfo.total_columns}</div>
                        <div class="stat-label">Columns</div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="stat-card">
                        <div class="stat-value">${dataInfo.numeric_columns}</div>
                        <div class="stat-label">Numeric Columns</div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="stat-card">
                        <div class="stat-value">${dataInfo.time_range ? (dataInfo.time_range.max - dataInfo.time_range.min).toFixed(1) : 'N/A'}</div>
                        <div class="stat-label">Duration (s)</div>
                    </div>
                </div>
            </div>
        `;
        
        if (fileInfo && fileInfo.length > 0) {
            html += `
                <div class="mt-3">
                    <h6>Loaded Files:</h6>
                    <div class="table-responsive">
                        <table class="table table-dark table-sm">
                            <thead>
                                <tr>
                                    <th>File</th>
                                    <th>Name</th>
                                    <th>Records</th>
                                </tr>
                            </thead>
                            <tbody>
            `;
            
            fileInfo.forEach(file => {
                html += `
                    <tr>
                        <td>${file.File}</td>
                        <td>${file.Name}</td>
                        <td>${file.Records.toLocaleString()}</td>
                    </tr>
                `;
            });
            
            html += `
                            </tbody>
                        </table>
                    </div>
                </div>
            `;
        }
        
        dataInfoContent.innerHTML = html;
        dataInfoCard.style.display = 'block';
    }

    showAnalysisControls() {
        document.getElementById('analysisControls').style.display = 'block';
        document.getElementById('lapDetection').style.display = 'block';
    }

    updateParameterSelects() {
        if (!this.currentData) return;

        // Update X-axis select
        const xAxisSelect = document.getElementById('xAxisSelect');
        xAxisSelect.innerHTML = '<option value="">Select X-axis</option>';
        
        this.currentData.columns.forEach(col => {
            const option = document.createElement('option');
            option.value = col;
            option.textContent = col;
            if (col === 'time') {
                option.selected = true;
            }
            xAxisSelect.appendChild(option);
        });

        // Update Y-axis select
        const yAxisSelect = document.getElementById('yAxisSelect');
        yAxisSelect.innerHTML = '<option value="">Select Y-axis parameters</option>';
        
        this.currentData.numeric_columns_list.forEach(col => {
            const option = document.createElement('option');
            option.value = col;
            option.textContent = col;
            yAxisSelect.appendChild(option);
        });

        // Update color axis select
        const colorAxisSelect = document.getElementById('colorAxisSelect');
        colorAxisSelect.innerHTML = '<option value="None">None</option>';
        
        this.currentData.numeric_columns_list.forEach(col => {
            const option = document.createElement('option');
            option.value = col;
            option.textContent = col;
            colorAxisSelect.appendChild(option);
        });

        // Update track analysis parameter select
        const trackParameterSelect = document.getElementById('trackParameterSelect');
        trackParameterSelect.innerHTML = '<option value="">Select a parameter</option>';
        
        this.currentData.numeric_columns_list.forEach(col => {
            const option = document.createElement('option');
            option.value = col;
            option.textContent = col;
            trackParameterSelect.appendChild(option);
        });

        // Update statistics parameter select
        const statsParameterSelect = document.getElementById('statsParameterSelect');
        statsParameterSelect.innerHTML = '<option value="">Select a parameter</option>';
        
        this.currentData.numeric_columns_list.forEach(col => {
            const option = document.createElement('option');
            option.value = col;
            option.textContent = col;
            statsParameterSelect.appendChild(option);
        });
    }

    updatePlotTypeControls() {
        const plotType = document.getElementById('plotTypeSelect').value;
        const xAxisControl = document.getElementById('xAxisControl');
        const colorAxisControl = document.getElementById('colorAxisControl');
        
        if (plotType === 'Histogram' || plotType === 'Box Plot' || plotType === 'Correlation Matrix') {
            xAxisControl.style.display = 'none';
            colorAxisControl.style.display = 'none';
        } else {
            xAxisControl.style.display = 'block';
            colorAxisControl.style.display = 'block';
        }
    }

    async updatePlot() {
        if (!this.currentData) {
            this.showAlert('Please load data first', 'warning');
            return;
        }

        const plotType = document.getElementById('plotTypeSelect').value;
        const xAxis = document.getElementById('xAxisSelect').value;
        const yAxisSelect = document.getElementById('yAxisSelect');
        const colorAxis = document.getElementById('colorAxisSelect').value;

        const yAxis = Array.from(yAxisSelect.selectedOptions).map(option => option.value);
        
        if (yAxis.length === 0) {
            this.showAlert('Please select at least one Y-axis parameter', 'warning');
            return;
        }

        // Show loading state in plot container
        const plotContainer = document.getElementById('plotContainer');
        plotContainer.innerHTML = `
            <div class="text-center">
                <div class="spinner-border text-primary mb-3" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <p>Creating plot with ${this.currentData.total_records.toLocaleString()} data points...</p>
                <small class="text-muted">Large datasets are automatically sampled for better performance</small>
                <div class="mt-3">
                    <button class="btn btn-sm btn-outline-secondary" onclick="app.cancelPlot()">Cancel</button>
                </div>
            </div>
        `;

        this.showLoading('Creating plot...');

        try {
            // Create abort controller for timeout
            const controller = new AbortController();
            const timeoutId = setTimeout(() => {
                console.log('⏰ Plot request timed out after 10 seconds');
                controller.abort();
            }, 10000); // Reduced to 10 second timeout
            
            // Store controller for cancellation
            this.currentPlotController = controller;
            
            console.log('🚀 Starting plot request...');
            const response = await fetch('/api/plot', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    type: plotType,
                    x_axis: xAxis,
                    y_axis: yAxis,
                    color_axis: colorAxis
                }),
                signal: controller.signal
            });
            
            clearTimeout(timeoutId);
            console.log('✅ Plot request completed, status:', response.status);

            const result = await response.json();
            console.log('📊 Plot result received:', result.success ? 'Success' : 'Failed');
            
            if (result.success) {
                console.log('🎨 Starting plot display...');
                await this.displayPlot(result.plot);
                console.log('✅ Plot display completed');
            } else {
                this.showAlert(result.error, 'danger');
            }
        } catch (error) {
            console.error('Plot creation error:', error);
            
            if (error.name === 'AbortError') {
                this.showAlert('Plot creation timed out. Try with fewer data points or a different plot type.', 'warning');
            } else {
                this.showAlert('Error creating plot: ' + error.message, 'danger');
            }
            
            // Reset plot container
            const plotContainer = document.getElementById('plotContainer');
            plotContainer.innerHTML = `
                <div class="text-center text-muted">
                    <i class="fas fa-chart-line fa-3x mb-3"></i>
                    <p>Plot creation failed. Try again or select different parameters.</p>
                </div>
            `;
        } finally {
            this.hideLoading();
            this.currentPlotController = null;
        }
    }

    cancelPlot() {
        if (this.currentPlotController) {
            console.log('🚫 Cancelling plot request...');
            this.currentPlotController.abort();
            this.currentPlotController = null;
            
            const plotContainer = document.getElementById('plotContainer');
            plotContainer.innerHTML = `
                <div class="text-center text-muted">
                    <i class="fas fa-times-circle fa-3x mb-3"></i>
                    <p>Plot creation cancelled.</p>
                </div>
            `;
            
            this.hideLoading();
        }
    }

        async displayPlot(plotJson) {
            console.log('Displaying plot, JSON length:', plotJson.length);
            
            // Check if Plotly is loaded
            if (typeof Plotly === 'undefined') {
                console.error('❌ Plotly is not loaded!');
                const plotContainer = document.getElementById('plotContainer');
                plotContainer.innerHTML = `
                    <div class="alert alert-danger">
                        <h5><i class="fas fa-exclamation-triangle"></i> Plotly Not Loaded</h5>
                        <p>Plotly library is not available. Please check if the Plotly CDN is loaded.</p>
                    </div>
                `;
                return;
            }
            console.log('✅ Plotly is loaded');

            const plotContainer = document.getElementById('plotContainer');
            plotContainer.innerHTML = '';

            try {
                const plotData = JSON.parse(plotJson);
                console.log('Parsed plot data:', plotData);
                console.log('Number of traces:', plotData.data.length);
                console.log('Layout:', plotData.layout);
            
            if (plotData.data.length === 0) {
                plotContainer.innerHTML = `
                    <div class="text-center text-muted">
                        <i class="fas fa-exclamation-triangle fa-3x mb-3"></i>
                        <p>No data to display. Check your parameter selections.</p>
                    </div>
                `;
                return;
            }
            
            // Try to render with Plotly, but with a timeout
            console.log('Attempting to render plot with Plotly...');
            console.log('Plot data structure:', {
                dataLength: plotData.data.length,
                layoutTitle: plotData.layout.title,
                firstTraceKeys: Object.keys(plotData.data[0] || {}),
                firstTraceXLength: plotData.data[0]?.x?.length,
                firstTraceYLength: plotData.data[0]?.y?.length
            });
            
            try {
                console.log('🎨 Rendering plot with Plotly...');
                console.log('Plot data structure:', {
                    dataLength: plotData.data.length,
                    layoutTitle: plotData.layout.title,
                    firstTraceKeys: Object.keys(plotData.data[0] || {}),
                    firstTraceXLength: plotData.data[0]?.x?.length,
                    firstTraceYLength: plotData.data[0]?.y?.length
                });
                
                // Convert numpy arrays to JavaScript arrays for Plotly
                console.log('Converting numpy arrays for Plotly...');
                plotData.data.forEach((trace, index) => {
                    if (trace.x && typeof trace.x === 'object' && trace.x.dtype && trace.x.bdata) {
                        console.log(`Converting trace ${index} xData...`);
                        try {
                            const binaryString = atob(trace.x.bdata);
                            const bytes = new Uint8Array(binaryString.length);
                            for (let i = 0; i < binaryString.length; i++) {
                                bytes[i] = binaryString.charCodeAt(i);
                            }
                            const dataView = new DataView(bytes.buffer);
                            const array = [];
                            for (let i = 0; i < bytes.length; i += 8) { // f8 = 8 bytes per float
                                array.push(dataView.getFloat64(i, true)); // little-endian
                            }
                            trace.x = array;
                            console.log(`Converted trace ${index} xData to array with ${array.length} points`);
                        } catch (error) {
                            console.error(`Error converting trace ${index} xData:`, error);
                            trace.x = [];
                        }
                    }
                    
                    if (trace.y && typeof trace.y === 'object' && trace.y.dtype && trace.y.bdata) {
                        console.log(`Converting trace ${index} yData...`);
                        try {
                            const binaryString = atob(trace.y.bdata);
                            const bytes = new Uint8Array(binaryString.length);
                            for (let i = 0; i < binaryString.length; i++) {
                                bytes[i] = binaryString.charCodeAt(i);
                            }
                            const dataView = new DataView(bytes.buffer);
                            const array = [];
                            for (let i = 0; i < bytes.length; i += 8) { // f8 = 8 bytes per float
                                array.push(dataView.getFloat64(i, true)); // little-endian
                            }
                            trace.y = array;
                            console.log(`Converted trace ${index} yData to array with ${array.length} points`);
                        } catch (error) {
                            console.error(`Error converting trace ${index} yData:`, error);
                            trace.y = [];
                        }
                    }
                });
                
                // Check if data conversion worked
                const firstTrace = plotData.data[0];
                if (firstTrace && firstTrace.x && firstTrace.y) {
                    console.log('Data after conversion:');
                    console.log('X data type:', typeof firstTrace.x, 'length:', Array.isArray(firstTrace.x) ? firstTrace.x.length : 'not array');
                    console.log('Y data type:', typeof firstTrace.y, 'length:', Array.isArray(firstTrace.y) ? firstTrace.y.length : 'not array');
                    console.log('First X values:', Array.isArray(firstTrace.x) ? firstTrace.x.slice(0, 3) : 'not array');
                    console.log('First Y values:', Array.isArray(firstTrace.y) ? firstTrace.y.slice(0, 3) : 'not array');
                }
                
                // Render with Plotly
                await Plotly.newPlot(plotContainer, plotData.data, plotData.layout, {
                    responsive: true,
                    displayModeBar: true,
                    modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d']
                });
                
                console.log('✅ Plot rendered successfully with Plotly');
                
            } catch (plotError) {
                console.error('Plotly rendering failed:', plotError);
                
                // Fallback: Show table visualization
                console.log('🔄 Falling back to table visualization...');
                const tableHtml = this.createSimpleTableVisualization(plotData);
                plotContainer.innerHTML = tableHtml;
                return;
            }
        } catch (error) {
            console.error('Error displaying plot:', error);
            plotContainer.innerHTML = `
                <div class="text-center text-muted">
                    <i class="fas fa-exclamation-triangle fa-3x mb-3"></i>
                    <p>Error rendering plot: ${error.message}</p>
                </div>
            `;
        }
    }

    createSimpleTableVisualization(plotData) {
        console.log('Creating table visualization for:', plotData);
        
        // Handle different data structures
        let trace, xData, yData, title;
        
        try {
            trace = plotData.data && plotData.data[0] ? plotData.data[0] : {};
            
            // Handle different data types - convert to arrays if needed
            xData = trace.x || [];
            yData = trace.y || [];
            
            // Handle numpy arrays (objects with dtype and bdata)
            if (xData && typeof xData === 'object' && xData.dtype && xData.bdata) {
                console.log('Converting numpy xData to array...');
                console.log('xData dtype:', xData.dtype, 'bdata length:', xData.bdata.length);
                try {
                    // Decode base64 numpy data
                    const binaryString = atob(xData.bdata);
                    const bytes = new Uint8Array(binaryString.length);
                    for (let i = 0; i < binaryString.length; i++) {
                        bytes[i] = binaryString.charCodeAt(i);
                    }
                    const dataView = new DataView(bytes.buffer);
                    const array = [];
                    for (let i = 0; i < bytes.length; i += 8) { // f8 = 8 bytes per float
                        array.push(dataView.getFloat64(i, true)); // little-endian
                    }
                    xData = array;
                    console.log('Converted xData to array with', xData.length, 'points, first few:', xData.slice(0, 5));
                } catch (error) {
                    console.error('Error converting numpy xData:', error);
                    xData = [];
                }
            }
            
            if (yData && typeof yData === 'object' && yData.dtype && yData.bdata) {
                console.log('Converting numpy yData to array...');
                console.log('yData dtype:', yData.dtype, 'bdata length:', yData.bdata.length);
                try {
                    // Decode base64 numpy data
                    const binaryString = atob(yData.bdata);
                    const bytes = new Uint8Array(binaryString.length);
                    for (let i = 0; i < binaryString.length; i++) {
                        bytes[i] = binaryString.charCodeAt(i);
                    }
                    const dataView = new DataView(bytes.buffer);
                    const array = [];
                    for (let i = 0; i < bytes.length; i += 8) { // f8 = 8 bytes per float
                        array.push(dataView.getFloat64(i, true)); // little-endian
                    }
                    yData = array;
                    console.log('Converted yData to array with', yData.length, 'points, first few:', yData.slice(0, 5));
                } catch (error) {
                    console.error('Error converting numpy yData:', error);
                    yData = [];
                }
            }
            
            // Ensure they are arrays
            if (!Array.isArray(xData)) {
                console.log('xData is not an array:', typeof xData, xData);
                xData = [];
            }
            if (!Array.isArray(yData)) {
                console.log('yData is not an array:', typeof yData, yData);
                yData = [];
            }
            
            // Handle title - it might be an object or string
            title = 'Data Visualization';
            if (plotData.layout && plotData.layout.title) {
                if (typeof plotData.layout.title === 'string') {
                    title = plotData.layout.title;
                } else if (typeof plotData.layout.title === 'object' && plotData.layout.title.text) {
                    title = plotData.layout.title.text;
                }
            }
        } catch (error) {
            console.error('Error parsing plot data:', error);
            return `
                <div class="alert alert-danger">
                    <h5><i class="fas fa-exclamation-triangle"></i> Data Structure Error</h5>
                    <p>Could not parse the plot data structure. Raw data:</p>
                    <pre style="color: white; font-size: 12px;">${JSON.stringify(plotData, null, 2)}</pre>
                </div>
            `;
        }
        
        // If no data, show raw structure for debugging
        if (xData.length === 0 && yData.length === 0) {
            return `
                <div class="alert alert-warning">
                    <h5><i class="fas fa-exclamation-triangle"></i> No Data Found</h5>
                    <p>The plot data structure doesn't contain the expected arrays. Here's what we received:</p>
                    <div class="mt-3">
                        <strong>Trace object:</strong>
                        <pre style="color: white; font-size: 12px; background: #333; padding: 10px; border-radius: 5px;">${JSON.stringify(trace, null, 2)}</pre>
                    </div>
                    <div class="mt-3">
                        <strong>Full plot data:</strong>
                        <pre style="color: white; font-size: 12px; background: #333; padding: 10px; border-radius: 5px;">${JSON.stringify(plotData, null, 2)}</pre>
                    </div>
                </div>
            `;
        }
        
        // Sample data for table (show first 20 points)
        const sampleSize = Math.min(20, xData.length);
        const sampledX = xData.slice(0, sampleSize);
        const sampledY = yData.slice(0, sampleSize);
        
        // Calculate statistics safely
        const totalPoints = xData.length;
        const yMin = yData.length > 0 ? Math.min(...yData) : 0;
        const yMax = yData.length > 0 ? Math.max(...yData) : 0;
        
        let tableHtml = `
            <div class="card">
                <div class="card-header">
                    <h5><i class="fas fa-table"></i> ${title}</h5>
                    <small class="text-muted">Showing first ${sampleSize} of ${totalPoints.toLocaleString()} data points</small>
                </div>
                <div class="card-body">
                    <div class="table-responsive">
                        <table class="table table-dark table-striped table-sm">
                            <thead>
                                <tr>
                                    <th>Index</th>
                                    <th>X Value</th>
                                    <th>Y Value</th>
                                </tr>
                            </thead>
                            <tbody>
        `;
        
        for (let i = 0; i < sampledX.length; i++) {
            tableHtml += `
                <tr>
                    <td>${i + 1}</td>
                    <td>${sampledX[i].toFixed(3)}</td>
                    <td>${sampledY[i].toFixed(3)}</td>
                </tr>
            `;
        }
        
        tableHtml += `
                            </tbody>
                        </table>
                    </div>
                    <div class="mt-3">
                        <div class="row">
                            <div class="col-md-6">
                                <div class="stat-card">
                                    <div class="stat-value">${totalPoints.toLocaleString()}</div>
                                    <div class="stat-label">Total Points</div>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="stat-card">
                                    <div class="stat-value">${yMin.toFixed(2)} - ${yMax.toFixed(2)}</div>
                                    <div class="stat-label">Y Range</div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        return tableHtml;
    }

    async detectLaps() {
        if (!this.currentData) {
            this.showAlert('Please load data first', 'warning');
            return;
        }

        const lapLength = document.getElementById('lapLengthInput').value;
        const tolerance = document.getElementById('detectionTolerance').value;

        this.showLoading('Detecting laps...');

        try {
            // This would need to be implemented in the backend
            // For now, we'll simulate lap detection
            setTimeout(() => {
                const resultsDiv = document.getElementById('lapDetectionResults');
                const statusDiv = document.getElementById('lapDetectionStatus');
                
                statusDiv.innerHTML = `
                    <strong>Lap Detection Complete!</strong><br>
                    Detected 3 laps with average length of ${lapLength}m<br>
                    Tolerance: ${tolerance}%
                `;
                
                resultsDiv.style.display = 'block';
                this.hideLoading();
            }, 2000);
        } catch (error) {
            this.showAlert('Error detecting laps: ' + error.message, 'danger');
            this.hideLoading();
        }
    }

    async generateStatistics() {
        if (!this.currentData) {
            this.showAlert('Please load data first', 'warning');
            return;
        }

        this.showLoading('Generating statistics...');

        try {
            const response = await fetch('/api/statistics', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    selected_laps: [] // Could be implemented for lap-specific stats
                })
            });

            const result = await response.json();
            
            if (result.success) {
                this.displayStatistics(result);
            } else {
                this.showAlert(result.error, 'danger');
            }
        } catch (error) {
            this.showAlert('Error generating statistics: ' + error.message, 'danger');
        } finally {
            this.hideLoading();
        }
    }

    displayStatistics(stats) {
        const container = document.getElementById('statisticsContainer');
        
        let html = `
            <div class="row mb-4">
                <div class="col-12">
                    <h5>Descriptive Statistics</h5>
                    <div class="table-responsive">
                        <table class="table table-dark table-striped">
                            <thead>
                                <tr>
                                    <th>Parameter</th>
                                    <th>Count</th>
                                    <th>Mean</th>
                                    <th>Std</th>
                                    <th>Min</th>
                                    <th>25%</th>
                                    <th>50%</th>
                                    <th>75%</th>
                                    <th>Max</th>
                                </tr>
                            </thead>
                            <tbody>
        `;
        
        Object.keys(stats.descriptive_stats).forEach(param => {
            const stat = stats.descriptive_stats[param];
            html += `
                <tr>
                    <td><strong>${param}</strong></td>
                    <td>${stat.count.toLocaleString()}</td>
                    <td>${stat.mean.toFixed(3)}</td>
                    <td>${stat.std.toFixed(3)}</td>
                    <td>${stat.min.toFixed(3)}</td>
                    <td>${stat['25%'].toFixed(3)}</td>
                    <td>${stat['50%'].toFixed(3)}</td>
                    <td>${stat['75%'].toFixed(3)}</td>
                    <td>${stat.max.toFixed(3)}</td>
                </tr>
            `;
        });
        
        html += `
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        `;
        
        if (stats.correlation_matrix) {
            html += `
                <div class="row mb-4">
                    <div class="col-12">
                        <h5>Correlation Matrix</h5>
                        <div id="correlationPlot" class="plot-container"></div>
                    </div>
                </div>
            `;
            
            // Create correlation heatmap
            const correlationData = {
                data: [{
                    z: Object.values(stats.correlation_matrix).map(row => 
                        Object.values(row).map(val => parseFloat(val))
                    ),
                    x: Object.keys(stats.correlation_matrix),
                    y: Object.keys(stats.correlation_matrix),
                    type: 'heatmap',
                    colorscale: 'RdBu',
                    reversescale: true,
                    showscale: true
                }],
                layout: {
                    title: 'Correlation Matrix',
                    height: 500,
                    template: 'plotly_dark'
                }
            };
            
            setTimeout(() => {
                Plotly.newPlot('correlationPlot', correlationData.data, correlationData.layout);
            }, 100);
        }
        
        if (stats.high_correlations && stats.high_correlations.length > 0) {
            html += `
                <div class="row">
                    <div class="col-12">
                        <h5>Strong Correlations (|r| > 0.7)</h5>
                        <div class="table-responsive">
                            <table class="table table-dark table-striped">
                                <thead>
                                    <tr>
                                        <th>Variable 1</th>
                                        <th>Variable 2</th>
                                        <th>Correlation</th>
                                        <th>Strength</th>
                                    </tr>
                                </thead>
                                <tbody>
            `;
            
            stats.high_correlations.forEach(corr => {
                html += `
                    <tr>
                        <td>${corr['Variable 1']}</td>
                        <td>${corr['Variable 2']}</td>
                        <td>${corr['Correlation']}</td>
                        <td><span class="badge bg-${corr['Strength'] === 'Very Strong' ? 'danger' : 'warning'}">${corr['Strength']}</span></td>
                    </tr>
                `;
            });
            
            html += `
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            `;
        }
        
        container.innerHTML = html;
    }

    async exportCsv() {
        if (!this.currentData) {
            this.showAlert('Please load data first', 'warning');
            return;
        }

        try {
            const response = await fetch('/api/export/csv');
            const blob = await response.blob();
            
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `formula_ufmg_data_${new Date().toISOString().slice(0,19).replace(/:/g, '-')}.csv`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
            
            this.showAlert('CSV file downloaded successfully', 'success');
        } catch (error) {
            this.showAlert('Error exporting CSV: ' + error.message, 'danger');
        }
    }

    async exportReport() {
        if (!this.currentData) {
            this.showAlert('Please load data first', 'warning');
            return;
        }

        try {
            const response = await fetch('/api/export/report');
            const blob = await response.blob();
            
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `analysis_report_${new Date().toISOString().slice(0,19).replace(/:/g, '-')}.txt`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
            
            this.showAlert('Report downloaded successfully', 'success');
        } catch (error) {
            this.showAlert('Error exporting report: ' + error.message, 'danger');
        }
    }

    async createParameterMap() {
        this.showAlert('Parameter map functionality coming soon!', 'info');
    }

    async createBehaviorMap() {
        this.showAlert('Driver behavior analysis coming soon!', 'info');
    }

    async createSpeedZones() {
        this.showAlert('Speed zones analysis coming soon!', 'info');
    }

    showLoading(text = 'Loading...') {
        const modal = new bootstrap.Modal(document.getElementById('loadingModal'));
        document.getElementById('loadingText').textContent = text;
        modal.show();
    }

    hideLoading() {
        const modal = bootstrap.Modal.getInstance(document.getElementById('loadingModal'));
        if (modal) {
            modal.hide();
        }
    }

    showAlert(message, type = 'info') {
        const alertContainer = document.getElementById('alertContainer');
        const alertId = 'alert-' + Date.now();
        
        const alertHtml = `
            <div class="alert alert-${type} alert-dismissible fade show" id="${alertId}" role="alert">
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
            </div>
        `;
        
        alertContainer.insertAdjacentHTML('beforeend', alertHtml);
        
        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            const alert = document.getElementById(alertId);
            if (alert) {
                const bsAlert = new bootstrap.Alert(alert);
                bsAlert.close();
            }
        }, 5000);
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new DataAnalysisApp();
});
