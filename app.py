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
        .content-wrapper {
            padding: 40px;
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
            padding: 20px;
            border-radius: 10px;
            margin: 15px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border-left: 4px solid #667eea;
        }
        .loss-function-card h5 {
            color: #667eea;
            margin-bottom: 8px;
            font-size: 1.1em;
        }
        .loss-function-card p {
            color: #666;
            line-height: 1.6;
            font-size: 0.95em;
        }
        .footer {
            background: linear-gradient(135deg, #2d3436 0%, #000000 100%);
            color: white;
            padding: 30px;
            text-align: center;
            margin-top: 50px;
            border-radius: 0 0 15px 15px;
        }
        .footer p {
            margin: 10px 0;
            line-height: 1.8;
        }
        .footer a {
            color: #667eea;
            text-decoration: none;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .footer a:hover {
            color: #764ba2;
            text-decoration: underline;
        }
        .footer .copyright-notice {
            font-size: 0.9em;
            opacity: 0.9;
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid rgba(255,255,255,0.2);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header-banner">
            <h1>🧪 ML Loss Function Playground</h1>
            <p>Interactive Educational Tool for Understanding Machine Learning Loss Functions</p>
        </div>
        
        <div class="content-wrapper">
            <div class="info-section">
                <h3>📚 What You'll Learn</h3>
                <ul>
                    <li><strong>Loss Function Comparison:</strong> See how different loss functions perform on the same dataset</li>
                    <li><strong>Convergence Behavior:</strong> Understand which loss functions converge faster for different tasks</li>
                    <li><strong>Training Dynamics:</strong> Visualize how training and validation losses evolve over epochs</li>
                    <li><strong>Robustness Testing:</strong> Experiment with noise, outliers, and mismatched loss functions</li>
                    <li><strong>Performance Metrics:</strong> Compare accuracy, precision, recall, F1-score for classification or MSE, MAE, R² for regression</li>
                </ul>
            </div>
            
            <div class="educational-info">
                <h4>💡 Educational Tips</h4>
                <p>Try intentionally mismatching loss functions with tasks (e.g., using MSE for classification) to observe:</p>
                <ul style="margin-top: 10px;">
                    <li>How training becomes unstable or slow</li>
                    <li>Why choosing the right loss function matters</li>
                    <li>The impact on convergence speed and final performance</li>
                </ul>
            </div>
            
            <h2 class="section-title">📖 Loss Functions Explained</h2>
            <div class="loss-function-card">
                <h5>🔹 MSE (Mean Squared Error)</h5>
                <p><strong>Best for:</strong> Regression tasks. Penalizes large errors heavily (squared term). Sensitive to outliers but provides smooth gradients for optimization.</p>
            </div>
            <div class="loss-function-card">
                <h5>🔹 MAE (Mean Absolute Error)</h5>
                <p><strong>Best for:</strong> Regression tasks with outliers. More robust to outliers than MSE as it doesn't square the errors. Provides constant gradient magnitude.</p>
            </div>
            <div class="loss-function-card">
                <h5>🔹 Cross Entropy</h5>
                <p><strong>Best for:</strong> Classification tasks. Measures the difference between predicted probability distributions and true labels. Standard choice for neural network classifiers.</p>
            </div>
            <div class="loss-function-card">
                <h5>🔹 BCE (Binary Cross Entropy)</h5>
                <p><strong>Best for:</strong> Binary classification (2 classes). Specialized version of cross entropy optimized for binary outcomes with sigmoid activation.</p>
            </div>
            <div class="loss-function-card">
                <h5>🔹 Huber Loss</h5>
                <p><strong>Best for:</strong> Regression with outliers. Combines benefits of MSE and MAE - quadratic for small errors, linear for large errors. Good balance of robustness and smooth gradients.</p>
            </div>
            
            <h2 class="section-title">⚙️ Configure Your Experiment</h2>
            <div class="warning">
                <strong>⚠️ Experimental Mode:</strong> This tool allows you to intentionally use "wrong" loss functions for educational purposes. Observe what happens when you use MSE for classification or Cross-Entropy for regression!
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
                <input type="number" id="noise_level" value="10" min="0" max="100" step="0.1" required>
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
                <label><input type="checkbox" value="mse"> MSE (Mean Squared Error)</label>
                <label><input type="checkbox" value="mae"> MAE (Mean Absolute Error)</label>
                <label><input type="checkbox" value="cross_entropy" checked> Cross Entropy</label>
                <label><input type="checkbox" value="sparse_categorical_crossentropy"> Sparse Categorical Cross Entropy</label>
                <label><input type="checkbox" value="bce"> BCE (Binary Cross Entropy)</label>
                <label><input type="checkbox" value="huber"> Huber Loss</label>
            </div>
        </div>
        <button onclick="trainModels()" id="trainBtn">🚀 Train Models</button>
        <div id="loading"><h3>Training in progress... ⏳</h3></div>
        <div id="results"></div>
    </div>
    <script>
        function updateLossOptions(){
            const taskType=document.getElementById('task_type').value;
            const checkboxes=document.querySelectorAll('#loss_functions input');
            checkboxes.forEach(cb=>cb.checked=false);
            document.getElementById('n_classes_group').style.display=(taskType==='multi_classification')?'block':'none';
            if(taskType==='regression'){
                document.querySelector('input[value="mse"]').checked=true;
            }else{
                document.querySelector('input[value="cross_entropy"]').checked=true;
                document.querySelector('input[value="sparse_categorical_crossentropy"]').checked=true;
                if(taskType==='binary_classification'){
                    document.querySelector('input[value="bce"]').checked=true;
                }
            }
        }
        async function trainModels(){
            const taskType=document.getElementById('task_type').value;
            const nSamples=parseInt(document.getElementById('n_samples').value);
            const nFeatures=parseInt(document.getElementById('n_features').value);
            const noiseLevelInput=document.getElementById('noise_level').value;
            const noiseLevel=parseFloat(noiseLevelInput);
            const outlierPercentage=parseFloat(document.getElementById('outlier_percentage').value);
            const outlierMagnitude=parseFloat(document.getElementById('outlier_magnitude').value);
            const nHiddenLayers=parseInt(document.getElementById('n_hidden_layers').value);
            const neuronsPerLayer=parseInt(document.getElementById('neurons_per_layer').value);
            const epochs=parseInt(document.getElementById('epochs').value);
            let nClasses=null;
            if(taskType==='multi_classification'){
                nClasses=parseInt(document.getElementById('n_classes').value);
            }
            const lossFunctions=Array.from(document.querySelectorAll('#loss_functions input:checked')).map(cb=>cb.value);
            if(lossFunctions.length===0){
                alert('Please select at least one loss function');
                return;
            }
            if(isNaN(noiseLevel)||noiseLevel<0||noiseLevel>100){
                document.getElementById('results').innerHTML=`<div class="error-message">Error: Noise level must be a number between 0 and 100</div>`;
                return;
            }
            if(isNaN(nSamples)||nSamples<100||nSamples>10000){
                document.getElementById('results').innerHTML=`<div class="error-message">Error: Number of samples must be an integer between 100 and 10000</div>`;
                return;
            }
            if(isNaN(nFeatures)||nFeatures<2||nFeatures>50){
                document.getElementById('results').innerHTML=`<div class="error-message">Error: Number of features must be an integer between 2 and 50</div>`;
                return;
            }
            if(isNaN(outlierPercentage)||outlierPercentage<0||outlierPercentage>20){
                document.getElementById('results').innerHTML=`<div class="error-message">Error: Outlier percentage must be a number between 0 and 20</div>`;
                return;
            }
            if(isNaN(outlierMagnitude)||outlierMagnitude<1||outlierMagnitude>100){
                document.getElementById('results').innerHTML=`<div class="error-message">Error: Outlier magnitude must be a number between 1 and 100</div>`;
                return;
            }
            if(isNaN(nHiddenLayers)||nHiddenLayers<1||nHiddenLayers>5){
                document.getElementById('results').innerHTML=`<div class="error-message">Error: Number of hidden layers must be an integer between 1 and 5</div>`;
                return;
            }
            if(isNaN(neuronsPerLayer)||neuronsPerLayer<8||neuronsPerLayer>128){
                document.getElementById('results').innerHTML=`<div class="error-message">Error: Neurons per layer must be an integer between 8 and 128</div>`;
                return;
            }
            if(isNaN(epochs)||epochs<10||epochs>500){
                document.getElementById('results').innerHTML=`<div class="error-message">Error: Epochs must be an integer between 10 and 500</div>`;
                return;
            }
            if(taskType==='multi_classification'&&(isNaN(nClasses)||nClasses<3||nClasses>10)){
                document.getElementById('results').innerHTML=`<div class="error-message">Error: Number of classes must be an integer between 3 and 10 for multi-classification</div>`;
                return;
            }
            document.getElementById('trainBtn').disabled=true;
            document.getElementById('loading').style.display='block';
            document.getElementById('results').innerHTML='';
            try{
                const response=await fetch('/train',{
                    method:'POST',
                    headers:{'Content-Type':'application/json'},
                    body:JSON.stringify({
                        task_type:taskType,
                        n_samples:nSamples,
                        n_features:nFeatures,
                        noise_level:noiseLevel,
                        outlier_percentage:outlierPercentage,
                        outlier_magnitude:outlierMagnitude,
                        n_hidden_layers:nHiddenLayers,
                        neurons_per_layer:neuronsPerLayer,
                        n_classes:nClasses,
                        loss_functions:lossFunctions,
                        epochs:epochs
                    })
                });
                if(!response.ok){
                    const errorText=await response.text();
                    throw new Error(`Server error: ${response.status} - ${errorText}`);
                }
                const data=await response.json();
                if(data.error){
                    throw new Error(data.error);
                }
                displayResults(data,epochs,taskType);
            }catch(error){
                document.getElementById('results').innerHTML=`<div class="error-message">Error: ${error.message}</div>`;
            }finally{
                document.getElementById('trainBtn').disabled=false;
                document.getElementById('loading').style.display='none';
            }
        }
        function displayResults(data,epochs,taskType){
            const resultsDiv=document.getElementById('results');
            let plotHTML='<h2>Loss Plots</h2>';
            Object.keys(data).forEach(lossName=>{
                plotHTML+=`
                    <div class="plot-container">
                        <h3>${lossName.toUpperCase()} Loss</h3>
                        <div id="lossPlot_${lossName}"></div>
                    </div>
                `;
            });
            plotHTML+=`
                <div class="plot-container">
                    <h2>Total Training Time</h2>
                    <div id="timePlot"></div>
                </div>
                <h2>Performance Metrics Summary</h2>
                <table class="summary-table" id="summaryTable"></table>
            `;
            resultsDiv.innerHTML=plotHTML;
            Object.keys(data).forEach(lossName=>{
                const traces=[
                    {
                        y:data[lossName].train_losses,
                        x:Array.from({length:epochs},(_,i)=>i+1),
                        name:'Train Loss',
                        mode:'lines',
                        line:{width:2}
                    },
                    {
                        y:data[lossName].val_losses,
                        x:Array.from({length:epochs},(_,i)=>i+1),
                        name:'Val Loss',
                        mode:'lines',
                        line:{width:2,dash:'dot'}
                    }
                ];
                Plotly.newPlot(`lossPlot_${lossName}`,traces,{
                    title:`${lossName.toUpperCase()} Train and Validation Loss Over Epochs`,
                    xaxis:{title:'Epoch'},
                    yaxis:{title:'Loss',type:'log'},
                    height:400
                });
            });
            const lossNames=Object.keys(data);
            const times=lossNames.map(lossName=>data[lossName].total_time);
            Plotly.newPlot('timePlot',[{
                x:lossNames,
                y:times,
                type:'bar',
                marker:{color:'rgb(158,202,225)',line:{color:'rgb(8,48,107)',width:1.5}}
            }],{
                title:'Total Training Time per Loss Function',
                xaxis:{title:'Loss Function'},
                yaxis:{title:'Time (seconds)'},
                height:400
            });
            let tableHTML='<tr><th>Loss Function</th><th>Convergence Epoch</th><th>Total Time (s)</th>';
            if(taskType==='regression'){
                tableHTML+='<th>MSE</th><th>MAE</th><th>R² Score</th></tr>';
            }else{
                tableHTML+='<th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1 Score</th></tr>';
            }
            const lossNamesArr=Object.keys(data);
            let bestMetrics={};
            if(taskType==='regression'){
                bestMetrics.mse=Math.min(...lossNamesArr.map(l=>data[l].metrics.mse));
                bestMetrics.mae=Math.min(...lossNamesArr.map(l=>data[l].metrics.mae));
                bestMetrics.r2=Math.max(...lossNamesArr.map(l=>data[l].metrics.r2));
            }else{
                bestMetrics.accuracy=Math.max(...lossNamesArr.map(l=>data[l].metrics.accuracy));
                bestMetrics.precision=Math.max(...lossNamesArr.map(l=>data[l].metrics.precision));
                bestMetrics.recall=Math.max(...lossNamesArr.map(l=>data[l].metrics.recall));
                bestMetrics.f1=Math.max(...lossNamesArr.map(l=>data[l].metrics.f1));
            }
            Object.keys(data).forEach(lossName=>{
                const result=data[lossName];
                const metrics=result.metrics;
                tableHTML+=`<tr>
                    <td><strong>${lossName}</strong></td>
                    <td>${result.convergence_epoch}</td>
                    <td>${result.total_time.toFixed(3)}</td>`;
                if(taskType==='regression'){
                    tableHTML+=`
                        <td class="${metrics.mse===bestMetrics.mse?'best-metric':''}">${metrics.mse.toFixed(4)}</td>
                        <td class="${metrics.mae===bestMetrics.mae?'best-metric':''}">${metrics.mae.toFixed(4)}</td>
                        <td class="${metrics.r2===bestMetrics.r2?'best-metric':''}">${metrics.r2.toFixed(4)}</td>
                    `;
                }else{
                    tableHTML+=`
                        <td class="${metrics.accuracy===bestMetrics.accuracy?'best-metric':''}">${metrics.accuracy.toFixed(4)}</td>
                        <td class="${metrics.precision===bestMetrics.precision?'best-metric':''}">${metrics.precision.toFixed(4)}</td>
                        <td class="${metrics.recall===bestMetrics.recall?'best-metric':''}">${metrics.recall.toFixed(4)}</td>
                        <td class="${metrics.f1===bestMetrics.f1?'best-metric':''}">${metrics.f1.toFixed(4)}</td>
                    `;
                }
                tableHTML+='</tr>';
            });
            document.getElementById('summaryTable').innerHTML=tableHTML;
        }
    </script>
</body>
</html>
'''


class SimpleNN(nn.Module):
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
    n_outliers = int(n_samples * (outlier_percentage / 100))
    if n_outliers == 0:
        return X, y
    outlier_indices = np.random.choice(n_samples, n_outliers, replace=False)
    X_outliers = X[outlier_indices].copy()
    X_outliers *= outlier_magnitude
    X[outlier_indices] = X_outliers
    return X, y


def clean_nans(data):
    if isinstance(data, list):
        return [None if isinstance(x, float) and (np.isnan(x) or np.isinf(x)) else x for x in data]
    elif isinstance(data, dict):
        return {k: clean_nans(v) for k, v in data.items()}
    return data


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/health')
def health():
    """Health check endpoint for Hugging Face Spaces"""
    return jsonify({'status': 'healthy'}), 200


@app.route('/train', methods=['POST'])
def train():
    try:
        config = request.get_json()
        if not config:
            logger.error("No JSON data received")
            return jsonify({'error': 'No JSON data received'}), 400
        
        logger.debug(f"Raw config received: {config}")
        
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
        
        if not all([task_type, n_samples, n_features, noise_level is not None, 
                   outlier_percentage is not None, outlier_magnitude, n_hidden_layers, 
                   neurons_per_layer, loss_functions, epochs]):
            logger.error("Missing required parameters")
            return jsonify({'error': 'Missing required parameters'}), 400
        
        if task_type not in ['regression', 'binary_classification', 'multi_classification']:
            logger.error(f"Invalid task_type: {task_type}")
            return jsonify({'error': 'Invalid task_type'}), 400
        
        if not isinstance(n_samples, int) or n_samples < 100 or n_samples > 10000:
            logger.error(f"Invalid n_samples: {n_samples}")
            return jsonify({'error': 'n_samples must be an integer between 100 and 10000'}), 400
        
        if not isinstance(n_features, int) or n_features < 2 or n_features > 50:
            logger.error(f"Invalid n_features: {n_features}")
            return jsonify({'error': 'n_features must be an integer between 2 and 50'}), 400
        
        try:
            noise_level = float(noise_level)
            if noise_level < 0 or noise_level > 100:
                logger.error(f"Invalid noise_level: {noise_level}")
                return jsonify({'error': 'noise_level must be a number between 0 and 100'}), 400
        except (TypeError, ValueError):
            logger.error(f"Invalid noise_level type or value: {noise_level}")
            return jsonify({'error': 'noise_level must be a valid number'}), 400
        
        try:
            outlier_percentage = float(outlier_percentage)
            if outlier_percentage < 0 or outlier_percentage > 20:
                logger.error(f"Invalid outlier_percentage: {outlier_percentage}")
                return jsonify({'error': 'outlier_percentage must be a number between 0 and 20'}), 400
        except (TypeError, ValueError):
            logger.error(f"Invalid outlier_percentage type or value: {outlier_percentage}")
            return jsonify({'error': 'outlier_percentage must be a valid number'}), 400
        
        try:
            outlier_magnitude = float(outlier_magnitude)
            if outlier_magnitude < 1 or outlier_magnitude > 100:
                logger.error(f"Invalid outlier_magnitude: {outlier_magnitude}")
                return jsonify({'error': 'outlier_magnitude must be a number between 1 and 100'}), 400
        except (TypeError, ValueError):
            logger.error(f"Invalid outlier_magnitude type or value: {outlier_magnitude}")
            return jsonify({'error': 'outlier_magnitude must be a valid number'}), 400
        
        if not isinstance(n_hidden_layers, int) or n_hidden_layers < 1 or n_hidden_layers > 5:
            logger.error(f"Invalid n_hidden_layers: {n_hidden_layers}")
            return jsonify({'error': 'n_hidden_layers must be an integer between 1 and 5'}), 400
        
        if not isinstance(neurons_per_layer, int) or neurons_per_layer < 8 or neurons_per_layer > 128:
            logger.error(f"Invalid neurons_per_layer: {neurons_per_layer}")
            return jsonify({'error': 'neurons_per_layer must be an integer between 8 and 128'}), 400
        
        if not isinstance(epochs, int) or epochs < 10 or epochs > 500:
            logger.error(f"Invalid epochs: {epochs}")
            return jsonify({'error': 'epochs must be an integer between 10 and 500'}), 400
        
        if task_type == 'multi_classification' and (not isinstance(n_classes, int) or n_classes < 3 or n_classes > 10):
            logger.error(f"Invalid n_classes: {n_classes}")
            return jsonify({'error': 'n_classes must be an integer between 3 and 10 for multi_classification'}), 400
        
        if not isinstance(loss_functions, list) or not loss_functions:
            logger.error(f"Invalid loss_functions: {loss_functions}")
            return jsonify({'error': 'loss_functions must be a non-empty list'}), 400
        
        valid_losses = ['mse', 'mae', 'cross_entropy', 'sparse_categorical_crossentropy', 'bce', 'huber']
        if not all(loss in valid_losses for loss in loss_functions):
            logger.error(f"Invalid loss functions: {loss_functions}")
            return jsonify({'error': f'loss_functions must be a subset of {valid_losses}'}), 400
        
        logger.debug(f"Validated config: {config}")
        
        if task_type == 'regression':
            X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise_level, random_state=42)
            output_dim = 1
            y = y.reshape(-1, 1)
        else:
            if task_type == 'binary_classification':
                n_classes = 2
            else:
                n_classes = n_classes
            flip_y = min(1.0, max(0.0, noise_level / 100))
            X, y = make_classification(n_samples=n_samples, n_features=n_features, 
                                      n_informative=max(2, n_features // 2), n_classes=n_classes, 
                                      n_clusters_per_class=1, random_state=42, flip_y=flip_y)
            output_dim = n_classes
        
        X, y = add_outliers(X, y, n_samples, outlier_percentage, outlier_magnitude)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        
        X_train_t = torch.FloatTensor(X_train)
        X_val_t = torch.FloatTensor(X_val)
        
        if task_type == 'regression':
            y_train_t = torch.FloatTensor(y_train)
            y_val_t = torch.FloatTensor(y_val)
        else:
            y_train_t = torch.LongTensor(y_train)
            y_val_t = torch.LongTensor(y_val)
        
        results = {}
        
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
                    logger.error(f"Error in training loss computation for {loss_name} at epoch {epoch}: {str(e)}")
                    loss = torch.tensor(0.0)
                
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
                        logger.error(f"Error in validation loss computation for {loss_name} at epoch {epoch}: {str(e)}")
                        val_loss = torch.tensor(0.0)
                
                train_losses.append(float(loss.item()) if not torch.isnan(loss) else 0.0)
                val_losses.append(float(val_loss.item()) if not torch.isnan(val_loss) else 0.0)
                
                if epoch > 10 and convergence_epoch is None and not np.isnan(val_losses[-1]):
                    recent_losses = val_losses[-5:]
                    if len(recent_losses) == 5:
                        loss_change = abs(recent_losses[-1] - recent_losses[0]) / (abs(recent_losses[0]) + 1e-8)
                        if loss_change < 0.001:
                            convergence_epoch = epoch
            
            total_time = time.time() - start_time
            
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
    # Get port from environment variable (Hugging Face uses PORT env var)
    port = int(os.environ.get('PORT', 7860))
    print("🚀 Starting ML Loss Function Playground...")
    print(f"📍 Server will be available on port {port}")
    print(f"🌐 Health check available at http://0.0.0.0:{port}/health")
    # Bind to 0.0.0.0 to allow external access, disable reloader for Hugging Face
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False, threaded=True)