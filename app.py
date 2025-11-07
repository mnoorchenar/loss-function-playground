from flask import Flask, render_template_string, request, jsonify
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
import time
import logging
import os

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>ML Loss Function Playground - Interactive Learning Tool</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .header-banner {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px 20px;
            text-align: center;
            border-radius: 15px 15px 0 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        .header-banner h1 {
            font-size: 2.5em;
            font-weight: 700;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }
        .header-banner p {
            font-size: 1.1em;
            opacity: 0.95;
            margin-top: 10px;
        }
        .container {
            max-width: 1600px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            overflow: hidden;
        }
        .tab-navigation {
            display: flex;
            background: #f8f9fa;
            border-bottom: 2px solid #e0e0e0;
        }
        .tab-button {
            flex: 1;
            padding: 20px;
            background: transparent;
            border: none;
            font-size: 1.1em;
            font-weight: 600;
            color: #666;
            cursor: pointer;
            transition: all 0.3s ease;
            border-bottom: 3px solid transparent;
        }
        .tab-button:hover {
            background: #fff;
            color: #667eea;
        }
        .tab-button.active {
            background: white;
            color: #667eea;
            border-bottom: 3px solid #667eea;
        }
        .tab-content {
            display: none;
            padding: 40px;
            max-height: 90vh;
            overflow-y: auto;
        }
        .tab-content.active {
            display: block;
        }
        .info-section {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 25px;
            border-radius: 12px;
            margin-bottom: 30px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .info-section h3 {
            font-size: 1.4em;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .info-section ul {
            list-style: none;
            padding-left: 0;
        }
        .info-section li {
            padding: 8px 0 8px 25px;
            position: relative;
            line-height: 1.6;
        }
        .info-section li:before {
            content: "✓";
            position: absolute;
            left: 0;
            font-weight: bold;
            color: #fff;
        }
        .section-title {
            font-size: 1.8em;
            color: #333;
            margin: 30px 0 20px 0;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
            font-weight: 600;
        }
        .controls {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 25px;
            margin-bottom: 30px;
        }
        .control-group {
            display: flex;
            flex-direction: column;
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            transition: all 0.3s ease;
            border: 2px solid transparent;
        }
        .control-group:hover {
            background: #fff;
            border-color: #667eea;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.15);
            transform: translateY(-2px);
        }
        label {
            font-weight: 600;
            margin-bottom: 8px;
            color: #333;
            font-size: 0.95em;
        }
        input, select {
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 15px;
            transition: all 0.3s ease;
            font-family: 'Inter', sans-serif;
        }
        input:focus, select:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        .checkbox-group {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 15px;
            background: #f8f9fa;
            padding: 25px;
            border-radius: 10px;
        }
        .checkbox-group label {
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 10px;
            cursor: pointer;
            padding: 10px;
            border-radius: 6px;
            transition: all 0.2s ease;
        }
        .checkbox-group label:hover {
            background: white;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .checkbox-group input[type="checkbox"] {
            width: 20px;
            height: 20px;
            cursor: pointer;
        }
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 18px 40px;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            font-size: 18px;
            font-weight: 600;
            margin-top: 30px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
            width: 100%;
            max-width: 400px;
            display: block;
            margin-left: auto;
            margin-right: auto;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5);
        }
        button:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }
        #results {
            margin-top: 40px;
        }
        .plot-container {
            margin-bottom: 40px;
            background: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        }
        .plot-container h3 {
            color: #667eea;
            margin-bottom: 20px;
            font-size: 1.4em;
        }
        #loading {
            display: none;
            text-align: center;
            padding: 40px;
            color: #667eea;
        }
        #loading h3 {
            font-size: 1.5em;
            animation: pulse 1.5s ease-in-out infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .summary-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            border-radius: 10px;
            overflow: hidden;
        }
        .summary-table th, .summary-table td {
            padding: 15px;
            text-align: left;
        }
        .summary-table th {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85em;
            letter-spacing: 0.5px;
        }
        .summary-table tr {
            border-bottom: 1px solid #eee;
            transition: all 0.2s ease;
        }
        .summary-table tr:hover {
            background: #f8f9fa;
        }
        .warning {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 20px 25px;
            border-radius: 10px;
            margin: 20px 0;
            box-shadow: 0 4px 15px rgba(240, 147, 251, 0.3);
        }
        .warning strong {
            font-size: 1.1em;
            display: block;
            margin-bottom: 8px;
        }
        .best-metric {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white;
            font-weight: 600;
            padding: 8px;
            border-radius: 6px;
        }
        .error-message {
            background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            box-shadow: 0 4px 15px rgba(235, 51, 73, 0.3);
        }
        .educational-info {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            padding: 25px;
            border-radius: 12px;
            margin: 20px 0;
            box-shadow: 0 4px 15px rgba(79, 172, 254, 0.3);
        }
        .educational-info h4 {
            font-size: 1.2em;
            margin-bottom: 10px;
        }
        .loss-function-card {
            background: white;
            padding: 25px;
            border-radius: 12px;
            margin: 25px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-left: 5px solid #667eea;
        }
        .loss-function-card h5 {
            color: #667eea;
            margin-bottom: 15px;
            font-size: 1.3em;
        }
        .loss-function-card p {
            color: #555;
            line-height: 1.8;
            font-size: 1em;
            margin-bottom: 10px;
        }
        .loss-function-card .formula {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin: 15px 0;
            overflow-x: auto;
            border-left: 4px solid #667eea;
        }
        .loss-function-card .use-case {
            background: #e8f5e9;
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
            border-left: 4px solid #4caf50;
        }
        .loss-function-card .use-case strong {
            color: #2e7d32;
        }
        .loss-function-card .pros {
            background: #e3f2fd;
            padding: 15px;
            border-radius: 8px;
            margin-top: 10px;
            border-left: 4px solid #2196f3;
        }
        .loss-function-card .cons {
            background: #fff3e0;
            padding: 15px;
            border-radius: 8px;
            margin-top: 10px;
            border-left: 4px solid #ff9800;
        }
        .concept-card {
            background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
            padding: 25px;
            border-radius: 12px;
            margin: 20px 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .concept-card h4 {
            color: #333;
            margin-bottom: 15px;
            font-size: 1.3em;
        }
        .concept-card p {
            color: #555;
            line-height: 1.8;
        }
        .metrics-explanation {
            background: #f5f5f5;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            border-left: 4px solid #667eea;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header-banner">
            <h1>🧪 ML Loss Function Playground</h1>
            <p>Interactive Educational Tool for Understanding Machine Learning Loss Functions</p>
            <p style="font-size: 0.9em; margin-top: 10px; opacity: 0.9;">Developed by Mohammad Noorchenarboo</p>
        </div>
        
        <div class="tab-navigation">
            <button class="tab-button active" onclick="switchTab('playground')">🎮 Playground</button>
            <button class="tab-button" onclick="switchTab('documentation')">📚 Documentation & Formulas</button>
        </div>
        
        <!-- PLAYGROUND TAB -->
        <div id="playground-tab" class="tab-content active">
            <div class="info-section">
                <h3>📚 What You'll Learn</h3>
                <ul>
                    <li><strong>Loss Function Comparison:</strong> See how different loss functions perform on the same dataset</li>
                    <li><strong>Convergence Behavior:</strong> Understand which loss functions converge faster</li>
                    <li><strong>Training Dynamics:</strong> Visualize how losses evolve over epochs</li>
                    <li><strong>Robustness Testing:</strong> Experiment with noise and outliers</li>
                    <li><strong>Performance Metrics:</strong> Compare classification or regression metrics</li>
                </ul>
            </div>
            
            <div class="educational-info">
                <h4>💡 Educational Tips</h4>
                <p>Try intentionally mismatching loss functions with tasks (e.g., using MSE for classification):</p>
                <ul style="margin-top: 10px; list-style: none; padding-left: 0;">
                    <li style="padding: 5px 0 5px 20px; position: relative;">
                        <span style="position: absolute; left: 0;">•</span> How training becomes unstable or slow
                    </li>
                    <li style="padding: 5px 0 5px 20px; position: relative;">
                        <span style="position: absolute; left: 0;">•</span> Why choosing the right loss function matters
                    </li>
                    <li style="padding: 5px 0 5px 20px; position: relative;">
                        <span style="position: absolute; left: 0;">•</span> The impact on convergence speed and performance
                    </li>
                </ul>
            </div>
            
            <h2 class="section-title">⚙️ Configure Your Experiment</h2>
            <div class="warning">
                <strong>⚠️ Experimental Mode:</strong> Intentionally use "wrong" loss functions for learning purposes!
            </div>
            
            <div class="controls">
                <div class="control-group">
                    <label>Task Type:</label>
                    <select id="task_type" onchange="updateLossOptions()">
                        <option value="regression">Regression</option>
                        <option value="binary_classification">Binary Classification</option>
                        <option value="multi_classification">Multi-class Classification</option>
                    </select>
                </div>
                <div class="control-group" id="n_classes_group" style="display:none;">
                    <label>Number of Classes:</label>
                    <input type="number" id="n_classes" value="3" min="3" max="10">
                </div>
                <div class="control-group">
                    <label>Number of Samples:</label>
                    <input type="number" id="n_samples" value="1000" min="100" max="10000">
                </div>
                <div class="control-group">
                    <label>Number of Features:</label>
                    <input type="number" id="n_features" value="10" min="2" max="50">
                </div>
                <div class="control-group">
                    <label>Noise Level (%):</label>
                    <input type="number" id="noise_level" value="10" min="0" max="100" step="0.1">
                </div>
                <div class="control-group">
                    <label>Outlier Percentage (%):</label>
                    <input type="number" id="outlier_percentage" value="1" min="0" max="20" step="0.1">
                </div>
                <div class="control-group">
                    <label>Outlier Magnitude:</label>
                    <input type="number" id="outlier_magnitude" value="10" min="1" max="100" step="1">
                </div>
                <div class="control-group">
                    <label>Number of Hidden Layers:</label>
                    <input type="number" id="n_hidden_layers" value="2" min="1" max="5">
                </div>
                <div class="control-group">
                    <label>Neurons per Hidden Layer:</label>
                    <input type="number" id="neurons_per_layer" value="32" min="8" max="128">
                </div>
                <div class="control-group">
                    <label>Epochs:</label>
                    <input type="number" id="epochs" value="100" min="10" max="500">
                </div>
            </div>
            <div class="control-group">
                <label>Loss Functions to Compare:</label>
                <div class="checkbox-group" id="loss_functions">
                    <label><input type="checkbox" value="mse"> MSE</label>
                    <label><input type="checkbox" value="mae"> MAE</label>
                    <label><input type="checkbox" value="cross_entropy" checked> Cross Entropy</label>
                    <label><input type="checkbox" value="sparse_categorical_crossentropy"> Sparse Cross Entropy</label>
                    <label><input type="checkbox" value="bce"> BCE</label>
                    <label><input type="checkbox" value="huber"> Huber Loss</label>
                </div>
            </div>
            <button onclick="trainModels()" id="trainBtn">🚀 Train Models</button>
            <div id="loading"><h3>Training in progress... ⏳</h3></div>
            <div id="results"></div>
        </div>

        <!-- DOCUMENTATION TAB -->
        <div id="documentation-tab" class="tab-content">
            <div class="info-section">
                <h3>📖 Loss Functions Theory & Mathematical Foundations</h3>
                <ul>
                    <li>Comprehensive mathematical explanations with LaTeX formulas</li>
                    <li>Use cases and practical applications</li>
                    <li>Pros and cons of different approaches</li>
                    <li>When to use each loss function</li>
                </ul>
            </div>

            <h2 class="section-title">Regression Loss Functions</h2>

            <div class="loss-function-card">
                <h5>🔹 Mean Squared Error (MSE)</h5>
                <p><strong>Primary Use:</strong> Regression tasks where large errors should be heavily penalized.</p>
                <div class="formula">
                    <strong>Mathematical Definition:</strong><br>
                    $$\\text{MSE} = \\frac{1}{n} \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2$$
                </div>
                <div class="pros">
                    <strong>✓ Advantages:</strong><ul style="margin-top: 8px; list-style: none; padding-left: 20px;">
                        <li>• Smooth, differentiable everywhere</li>
                        <li>• Strong gradient properties</li>
                        <li>• Penalizes large errors heavily</li>
                    </ul>
                </div>
                <div class="cons">
                    <strong>✗ Disadvantages:</strong><ul style="margin-top: 8px; list-style: none; padding-left: 20px;">
                        <li>• Very sensitive to outliers</li>
                        <li>• Can cause unstable training</li>
                    </ul>
                </div>
            </div>

            <div class="loss-function-card">
                <h5>🔹 Mean Absolute Error (MAE)</h5>
                <p><strong>Primary Use:</strong> Regression with outliers, where robustness is important.</p>
                <div class="formula">
                    <strong>Mathematical Definition:</strong><br>
                    $$\\text{MAE} = \\frac{1}{n} \\sum_{i=1}^{n} |y_i - \\hat{y}_i|$$
                </div>
                <div class="pros">
                    <strong>✓ Advantages:</strong><ul style="margin-top: 8px; list-style: none; padding-left: 20px;">
                        <li>• Robust to outliers</li>
                        <li>• Interpretable (same units as target)</li>
                        <li>• Handles extreme values</li>
                    </ul>
                </div>
                <div class="cons">
                    <strong>✗ Disadvantages:</strong><ul style="margin-top: 8px; list-style: none; padding-left: 20px;">
                        <li>• Non-differentiable at zero</li>
                        <li>• Slower convergence than MSE</li>
                    </ul>
                </div>
            </div>

            <div class="loss-function-card">
                <h5>🔹 Huber Loss</h5>
                <p><strong>Primary Use:</strong> Regression with outliers - combines best of MSE and MAE.</p>
                <div class="formula">
                    <strong>Mathematical Definition:</strong><br>
                    $$L_\\delta(y, \\hat{y}) = \\begin{cases} \\frac{1}{2}(y - \\hat{y})^2 & \\text{if } |y - \\hat{y}| \\leq \\delta \\\\ \\delta(|y - \\hat{y}| - \\frac{\\delta}{2}) & \\text{otherwise} \\end{cases}$$
                </div>
                <div class="pros">
                    <strong>✓ Advantages:</strong><ul style="margin-top: 8px; list-style: none; padding-left: 20px;">
                        <li>• Robust to outliers</li>
                        <li>• Smooth gradients</li>
                        <li>• Best of both worlds</li>
                    </ul>
                </div>
            </div>

            <h2 class="section-title">Classification Loss Functions</h2>

            <div class="loss-function-card">
                <h5>🔹 Binary Cross Entropy (BCE)</h5>
                <p><strong>Primary Use:</strong> Binary classification (2 classes).</p>
                <div class="formula">
                    <strong>Mathematical Definition:</strong><br>
                    $$\\text{BCE} = -\\frac{1}{n} \\sum_{i=1}^{n} [y_i \\log(\\hat{p}_i) + (1-y_i) \\log(1-\\hat{p}_i)]$$
                </div>
            </div>

            <div class="loss-function-card">
                <h5>🔹 Categorical Cross Entropy</h5>
                <p><strong>Primary Use:</strong> Multi-class classification.</p>
                <div class="formula">
                    <strong>Mathematical Definition:</strong><br>
                    $$\\text{CE} = -\\frac{1}{n} \\sum_{i=1}^{n} \\sum_{c=1}^{C} y_{i,c} \\log(\\hat{p}_{i,c})$$
                </div>
            </div>

            <h2 class="section-title">Choosing the Right Loss Function</h2>
            <div class="concept-card">
                <h4>🎯 Decision Tree</h4>
                <div style="margin-top: 15px; font-size: 0.95em; line-height: 2;">
                    Is it regression? → Use MSE (no outliers) or MAE (with outliers)<br>
                    Is it binary classification? → Use BCE<br>
                    Is it multi-class? → Use Cross Entropy
                </div>
            </div>
        </div>
    </div>

    <script>
        function switchTab(tabName) {
            document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
            document.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));
            document.getElementById(tabName + '-tab').classList.add('active');
            event.target.classList.add('active');
            if (tabName === 'documentation' && window.MathJax) {
                setTimeout(() => MathJax.typesetPromise(), 100);
            }
        }
        
        function updateLossOptions(){
            const taskType = document.getElementById('task_type').value;
            const checkboxes = document.querySelectorAll('#loss_functions input');
            checkboxes.forEach(cb => cb.checked = false);
            document.getElementById('n_classes_group').style.display = (taskType === 'multi_classification') ? 'block' : 'none';
            if (taskType === 'regression') {
                document.querySelector('input[value="mse"]').checked = true;
            } else {
                document.querySelector('input[value="cross_entropy"]').checked = true;
                document.querySelector('input[value="sparse_categorical_crossentropy"]').checked = true;
                if (taskType === 'binary_classification') {
                    document.querySelector('input[value="bce"]').checked = true;
                }
            }
        }
        
        async function trainModels(){
            const taskType = document.getElementById('task_type').value;
            const nSamples = parseInt(document.getElementById('n_samples').value);
            const nFeatures = parseInt(document.getElementById('n_features').value);
            const noiseLevel = parseFloat(document.getElementById('noise_level').value);
            const outlierPercentage = parseFloat(document.getElementById('outlier_percentage').value);
            const outlierMagnitude = parseFloat(document.getElementById('outlier_magnitude').value);
            const nHiddenLayers = parseInt(document.getElementById('n_hidden_layers').value);
            const neuronsPerLayer = parseInt(document.getElementById('neurons_per_layer').value);
            const epochs = parseInt(document.getElementById('epochs').value);
            let nClasses = null;
            if(taskType === 'multi_classification'){
                nClasses = parseInt(document.getElementById('n_classes').value);
            }
            const lossFunctions = Array.from(document.querySelectorAll('#loss_functions input:checked')).map(cb => cb.value);
            
            if(lossFunctions.length === 0){
                alert('Please select at least one loss function');
                return;
            }
            
            document.getElementById('trainBtn').disabled = true;
            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').innerHTML = '';
            
            try{
                const response = await fetch('/train', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        task_type: taskType,
                        n_samples: nSamples,
                        n_features: nFeatures,
                        noise_level: noiseLevel,
                        outlier_percentage: outlierPercentage,
                        outlier_magnitude: outlierMagnitude,
                        n_hidden_layers: nHiddenLayers,
                        neurons_per_layer: neuronsPerLayer,
                        n_classes: nClasses,
                        loss_functions: lossFunctions,
                        epochs: epochs
                    })
                });
                
                if(!response.ok){
                    throw new Error(`Server error: ${response.status}`);
                }
                
                const data = await response.json();
                if(data.error){
                    throw new Error(data.error);
                }
                displayResults(data, epochs, taskType);
            }catch(error){
                document.getElementById('results').innerHTML = `<div class="error-message">Error: ${error.message}</div>`;
            }finally{
                document.getElementById('trainBtn').disabled = false;
                document.getElementById('loading').style.display = 'none';
            }
        }
        
        function displayResults(data, epochs, taskType){
            const resultsDiv = document.getElementById('results');
            let plotHTML = '<h2 class="section-title">📈 Loss Plots</h2>';
            
            Object.keys(data).forEach(lossName => {
                plotHTML += `<div class="plot-container"><h3>${lossName.toUpperCase()}</h3><div id="lossPlot_${lossName}"></div></div>`;
            });
            
            plotHTML += `<div class="plot-container"><h2 class="section-title" style="margin-top: 0; border: none;">⏱️ Training Time</h2><div id="timePlot"></div></div>`;
            plotHTML += `<h2 class="section-title">📊 Performance Metrics</h2><table class="summary-table" id="summaryTable"></table>`;
            
            resultsDiv.innerHTML = plotHTML;
            
            Object.keys(data).forEach(lossName => {
                const traces = [
                    {y: data[lossName].train_losses, x: Array.from({length: epochs}, (_,i) => i+1), name: 'Train Loss', mode: 'lines', line: {width: 2}},
                    {y: data[lossName].val_losses, x: Array.from({length: epochs}, (_,i) => i+1), name: 'Val Loss', mode: 'lines', line: {width: 2, dash: 'dot'}}
                ];
                Plotly.newPlot(`lossPlot_${lossName}`, traces, {title: `${lossName.toUpperCase()} Loss`, xaxis: {title: 'Epoch'}, yaxis: {title: 'Loss', type: 'log'}, height: 400});
            });
            
            const lossNames = Object.keys(data);
            const times = lossNames.map(l => data[l].total_time);
            Plotly.newPlot('timePlot', [{x: lossNames, y: times, type: 'bar', marker: {color: 'rgb(158,202,225)', line: {color: 'rgb(8,48,107)', width: 1.5}}}], {title: 'Training Time', xaxis: {title: 'Loss Function'}, yaxis: {title: 'Time (s)'}, height: 400});
            
            let tableHTML = '<tr><th>Loss Function</th><th>Convergence Epoch</th><th>Total Time (s)</th>';
            if(taskType === 'regression'){
                tableHTML += '<th>MSE</th><th>MAE</th><th>R² Score</th></tr>';
            }else{
                tableHTML += '<th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1 Score</th></tr>';
            }
            
            const lossNamesArr = Object.keys(data);
            let bestMetrics = {};
            if(taskType === 'regression'){
                bestMetrics.mse = Math.min(...lossNamesArr.map(l => data[l].metrics.mse));
                bestMetrics.mae = Math.min(...lossNamesArr.map(l => data[l].metrics.mae));
                bestMetrics.r2 = Math.max(...lossNamesArr.map(l => data[l].metrics.r2));
            }else{
                bestMetrics.accuracy = Math.max(...lossNamesArr.map(l => data[l].metrics.accuracy));
                bestMetrics.precision = Math.max(...lossNamesArr.map(l => data[l].metrics.precision));
                bestMetrics.recall = Math.max(...lossNamesArr.map(l => data[l].metrics.recall));
                bestMetrics.f1 = Math.max(...lossNamesArr.map(l => data[l].metrics.f1));
            }
            
            Object.keys(data).forEach(lossName => {
                const result = data[lossName];
                const metrics = result.metrics;
                tableHTML += `<tr><td><strong>${lossName}</strong></td><td>${result.convergence_epoch}</td><td>${result.total_time.toFixed(3)}</td>`;
                if(taskType === 'regression'){
                    tableHTML += `<td class="${metrics.mse === bestMetrics.mse ? 'best-metric' : ''}">${metrics.mse.toFixed(4)}</td><td class="${metrics.mae === bestMetrics.mae ? 'best-metric' : ''}">${metrics.mae.toFixed(4)}</td><td class="${metrics.r2 === bestMetrics.r2 ? 'best-metric' : ''}">${metrics.r2.toFixed(4)}</td>`;
                }else{
                    tableHTML += `<td class="${metrics.accuracy === bestMetrics.accuracy ? 'best-metric' : ''}">${metrics.accuracy.toFixed(4)}</td><td class="${metrics.precision === bestMetrics.precision ? 'best-metric' : ''}">${metrics.precision.toFixed(4)}</td><td class="${metrics.recall === bestMetrics.recall ? 'best-metric' : ''}">${metrics.recall.toFixed(4)}</td><td class="${metrics.f1 === bestMetrics.f1 ? 'best-metric' : ''}">${metrics.f1.toFixed(4)}</td>`;
                }
                tableHTML += '</tr>';
            });
            
            document.getElementById('summaryTable').innerHTML = tableHTML;
        }
    </script>
</body>
</html>
'''


class SimpleNN(nn.Module):
    """Simple Neural Network for classification and regression tasks"""
    def __init__(self, input_dim, output_dim, task_type, loss_name, n_hidden_layers, neurons_per_layer):
        super(SimpleNN, self).__init__()
        self.task_type = task_type
        self.loss_name = loss_name
        
        layers = [nn.Linear(input_dim, neurons_per_layer), nn.ReLU()]
        for _ in range(n_hidden_layers - 1):
            layers.extend([nn.Linear(neurons_per_layer, neurons_per_layer), nn.ReLU()])
        layers.append(nn.Linear(neurons_per_layer, output_dim))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def get_loss_function(loss_name):
    """Returns appropriate loss function based on name"""
    losses = {
        'mse': nn.MSELoss(),
        'mae': nn.L1Loss(),
        'cross_entropy': nn.CrossEntropyLoss(),
        'sparse_categorical_crossentropy': nn.CrossEntropyLoss(),
        'bce': nn.BCEWithLogitsLoss(),
        'huber': nn.SmoothL1Loss()
    }
    return losses.get(loss_name, nn.MSELoss())


def evaluate_classification(model, X, y):
    """Evaluate classification model"""
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        if model.loss_name == 'bce' and model.task_type == 'binary_classification':
            predicted = (torch.sigmoid(outputs) > 0.5).long()
        else:
            _, predicted = torch.max(outputs, 1)
    
    y_true = y.cpu().numpy()
    y_pred = predicted.cpu().numpy()
    
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='macro', zero_division=0)
    }


def evaluate_regression(model, X, y):
    """Evaluate regression model"""
    model.eval()
    with torch.no_grad():
        outputs = model(X)
    
    y_true = y.cpu().numpy()
    y_pred = outputs.cpu().numpy()
    
    return {
        'mse': mean_squared_error(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }


def add_outliers(X, y, n_samples, outlier_percentage, outlier_magnitude):
    """Add outliers to dataset"""
    n_outliers = int(n_samples * (outlier_percentage / 100))
    if n_outliers == 0:
        return X, y
    
    outlier_indices = np.random.choice(n_samples, n_outliers, replace=False)
    X_outliers = X[outlier_indices].copy()
    X_outliers *= outlier_magnitude
    X[outlier_indices] = X_outliers
    
    return X, y


def clean_nans(data):
    """Clean NaN and Inf values from data"""
    if isinstance(data, list):
        return [None if isinstance(x, float) and (np.isnan(x) or np.isinf(x)) else x for x in data]
    elif isinstance(data, dict):
        return {k: clean_nans(v) for k, v in data.items()}
    return data


@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template_string(HTML_TEMPLATE)


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'}), 200


@app.route('/train', methods=['POST'])
def train():
    """Train models with different loss functions"""
    try:
        config = request.get_json()
        
        if not config:
            logger.error("No JSON data received")
            return jsonify({'error': 'No JSON data received'}), 400
        
        task_type = config.get('task_type')
        n_samples = config.get('n_samples')
        n_features = config.get('n_features')
        noise_level = config.get('noise_level')
        outlier_percentage = config.get('outlier_percentage')
        outlier_magnitude = config.get('outlier_magnitude')
        n_hidden_layers = config.get('n_hidden_layers')
        neurons_per_layer = config.get('neurons_per_layer')
        loss_functions = config.get('loss_functions')
        epochs = config.get('epochs')
        n_classes = config.get('n_classes')
        
        # Validate inputs
        if not all([task_type, n_samples, n_features, noise_level is not None, 
                   outlier_percentage is not None, outlier_magnitude, n_hidden_layers, 
                   neurons_per_layer, loss_functions, epochs]):
            return jsonify({'error': 'Missing required parameters'}), 400
        
        if task_type not in ['regression', 'binary_classification', 'multi_classification']:
            return jsonify({'error': 'Invalid task_type'}), 400
        
        # Generate dataset
        if task_type == 'regression':
            X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise_level, random_state=42)
            output_dim = 1
            y = y.reshape(-1, 1)
        else:
            if task_type == 'binary_classification':
                n_classes = 2
            else:
                n_classes = n_classes or 3
            
            flip_y = min(1.0, max(0.0, noise_level / 100))
            X, y = make_classification(
                n_samples=n_samples, 
                n_features=n_features, 
                n_informative=max(2, n_features // 2), 
                n_classes=n_classes, 
                n_clusters_per_class=1, 
                random_state=42, 
                flip_y=flip_y
            )
            output_dim = n_classes
        
        # Add outliers and split
        X, y = add_outliers(X, y, n_samples, outlier_percentage, outlier_magnitude)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Normalize
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train)
        X_val_t = torch.FloatTensor(X_val)
        
        if task_type == 'regression':
            y_train_t = torch.FloatTensor(y_train)
            y_val_t = torch.FloatTensor(y_val)
        else:
            y_train_t = torch.LongTensor(y_train)
            y_val_t = torch.LongTensor(y_val)
        
        results = {}
        
        # Train model for each loss function
        for loss_name in loss_functions:
            if task_type == 'binary_classification' and loss_name == 'bce':
                output_dim_loss = 1
            else:
                output_dim_loss = output_dim
            
            model = SimpleNN(n_features, output_dim_loss, task_type, loss_name, n_hidden_layers, neurons_per_layer)
            criterion = get_loss_function(loss_name)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            train_losses = []
            val_losses = []
            start_time = time.time()
            convergence_epoch = None
            
            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                outputs = model(X_train_t)
                
                try:
                    if task_type != 'regression' and loss_name in ['cross_entropy', 'sparse_categorical_crossentropy']:
                        loss = criterion(outputs, y_train_t)
                    elif task_type == 'binary_classification' and loss_name == 'bce':
                        loss = criterion(outputs, y_train_t.float())
                    else:
                        if task_type != 'regression':
                            loss = criterion(outputs.mean(dim=1, keepdim=True), y_train_t.float().unsqueeze(1))
                        else:
                            loss = criterion(outputs, y_train_t)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                except Exception as e:
                    logger.error(f"Error in training for {loss_name} at epoch {epoch}: {str(e)}")
                    loss = torch.tensor(0.0)
                
                # Validation
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val_t)
                    try:
                        if task_type != 'regression' and loss_name in ['cross_entropy', 'sparse_categorical_crossentropy']:
                            val_loss = criterion(val_outputs, y_val_t)
                        elif task_type == 'binary_classification' and loss_name == 'bce':
                            val_loss = criterion(val_outputs, y_val_t.float())
                        else:
                            if task_type != 'regression':
                                val_loss = criterion(val_outputs.mean(dim=1, keepdim=True), y_val_t.float().unsqueeze(1))
                            else:
                                val_loss = criterion(val_outputs, y_val_t)
                    except Exception as e:
                        logger.error(f"Error in validation for {loss_name} at epoch {epoch}: {str(e)}")
                        val_loss = torch.tensor(0.0)
                
                train_losses.append(float(loss.item()) if not torch.isnan(loss) else 0.0)
                val_losses.append(float(val_loss.item()) if not torch.isnan(val_loss) else 0.0)
                
                # Check convergence
                if epoch > 10 and convergence_epoch is None and not np.isnan(val_losses[-1]):
                    recent_losses = val_losses[-5:]
                    if len(recent_losses) == 5:
                        loss_change = abs(recent_losses[-1] - recent_losses[0]) / (abs(recent_losses[0]) + 1e-8)
                        if loss_change < 0.001:
                            convergence_epoch = epoch
            
            total_time = time.time() - start_time
            
            # Evaluate
            if task_type == 'regression':
                metrics = evaluate_regression(model, X_val_t, y_val_t)
            else:
                metrics = evaluate_classification(model, X_val_t, y_val_t)
            
            results[loss_name] = {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'total_time': total_time,
                'convergence_epoch': convergence_epoch if convergence_epoch else epochs,
                'metrics': metrics
            }
        
        logger.debug("Training completed successfully")
        return jsonify(clean_nans(results))
    
    except Exception as e:
        logger.error(f"Error in /train route: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    print("=" * 60)
    print("🚀 ML Loss Function Playground - Starting Server")
    print("=" * 60)
    print(f"📍 Server running on port {port}")
    print(f"🌐 Open http://localhost:{port} in your browser")
    print(f"📊 Health check: http://localhost:{port}/health")
    print("=" * 60)
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False, threaded=True)