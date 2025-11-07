# app.py
from flask import Flask, render_template_string, request, jsonify
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)
import torch
import torch.nn as nn
import torch.optim as optim
import time
import logging
import os

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

HTML_TEMPLATE = '''<!DOCTYPE html>
<html>
<head>
    <title>ML Loss Function Playground</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Inter', sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; padding: 20px; }
        .header-banner { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 40px 20px; text-align: center; border-radius: 15px 15px 0 0; box-shadow: 0 4px 15px rgba(0,0,0,0.2); }
        .header-banner h1 { font-size: 2.5em; font-weight: 700; margin-bottom: 10px; }
        .container { max-width: 1600px; margin: 0 auto; background: white; border-radius: 15px; box-shadow: 0 10px 40px rgba(0,0,0,0.2); overflow: hidden; }
        .tab-navigation { display: flex; background: #f8f9fa; border-bottom: 2px solid #e0e0e0; }
        .tab-button { flex: 1; padding: 20px; background: transparent; border: none; font-size: 1.1em; font-weight: 600; color: #666; cursor: pointer; transition: all 0.3s ease; border-bottom: 3px solid transparent; }
        .tab-button:hover { background: #fff; color: #667eea; }
        .tab-button.active { background: white; color: #667eea; border-bottom: 3px solid #667eea; }
        .tab-content { display: none; padding: 40px; }
        .tab-content.active { display: block; }
        .info-section { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 25px; border-radius: 12px; margin-bottom: 30px; }
        .info-section h3 { font-size: 1.4em; margin-bottom: 15px; }
        .section-title { font-size: 1.8em; color: #333; margin: 30px 0 20px; padding-bottom: 10px; border-bottom: 3px solid #667eea; }
        .controls { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 25px; margin-bottom: 30px; }
        .control-group { background: #f8f9fa; padding: 20px; border-radius: 10px; border: 2px solid transparent; }
        .control-group:hover { border-color: #667eea; }
        label { font-weight: 600; margin-bottom: 8px; display: block; }
        input, select { width: 100%; padding: 12px; border: 2px solid #e0e0e0; border-radius: 8px; font-size: 15px; }
        input:focus, select:focus { border-color: #667eea; outline: none; }
        .checkbox-group { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 15px; background: #f8f9fa; padding: 25px; border-radius: 10px; }
        .checkbox-group label { display: flex; align-items: center; gap: 10px; cursor: pointer; }
        button { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 18px 40px; border: none; border-radius: 10px; font-size: 18px; font-weight: 600; margin: 30px auto; display: block; width: fit-content; }
        button:disabled { background: #ccc; cursor: not-allowed; }
        #loading { display: none; text-align: center; padding: 40px; }
        .plot-container { margin: 40px 0; background: white; padding: 25px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); }
        .summary-table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        .summary-table th, .summary-table td { padding: 15px; text-align: left; border-bottom: 1px solid #eee; }
        .summary-table th { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
        .best-metric { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: white; font-weight: 600; padding: 8px; border-radius: 6px; }
        .loss-function-card { background: white; padding: 25px; border-radius: 12px; margin: 20px 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1); border-left: 5px solid #667eea; }
        .loss-function-card .formula { background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 15px 0; overflow-x: auto; font-family: 'Courier New', monospace; }
        .footer { background: #2d3436; color: white; padding: 30px; text-align: center; margin-top: 50px; border-radius: 0 0 15px 15px; }
    </style>
</head>
<body>
<div class="container">
    <div class="header-banner">
        <h1>ML Loss Function Playground</h1>
        <p>Interactive Tool to Compare Loss Functions</p>
        <p style="font-size: 0.9em; margin-top: 10px;">by Mohammad Noorchenarboo</p>
    </div>

    <div class="tab-navigation">
        <button class="tab-button active" onclick="switchTab('playground')">Playground</button>
        <button class="tab-button" onclick="switchTab('documentation')">Documentation & Formulas</button>
    </div>

    <!-- PLAYGROUND TAB -->
    <div id="playground-tab" class="tab-content active">
        <div class="info-section">
            <h3>What You'll Learn</h3>
            <ul>
                <li>Compare loss functions on the same dataset</li>
                <li>See convergence speed and stability</li>
                <li>Experiment with noise and outliers</li>
                <li>Try mismatched loss functions (educational!)</li>
            </ul>
        </div>

        <h2 class="section-title">Configure Experiment</h2>
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
                <label>Samples:</label>
                <input type="number" id="n_samples" value="1000" min="100" max="10000">
            </div>
            <div class="control-group">
                <label>Features:</label>
                <input type="number" id="n_features" value="10" min="2" max="50">
            </div>
            <div class="control-group">
                <label>Noise Level (%):</label>
                <input type="number" id="noise_level" value="10" min="0" max="100" step="0.1">
            </div>
            <div class="control-group">
                <label>Outlier %:</label>
                <input type="number" id="outlier_percentage" value="1" min="0" max="20" step="0.1">
            </div>
            <div class="control-group">
                <label>Outlier Magnitude:</label>
                <input type="number" id="outlier_magnitude" value="10" min="1" max="100">
            </div>
            <div class="control-group">
                <label>Hidden Layers:</label>
                <input type="number" id="n_hidden_layers" value="2" min="1" max="5">
            </div>
            <div class="control-group">
                <label>Neurons/Layer:</label>
                <input type="number" id="neurons_per_layer" value="32" min="8" max="128">
            </div>
            <div class="control-group">
                <label>Epochs:</label>
                <input type="number" id="epochs" value="100" min="10" max="500">
            </div>
        </div>

        <div class="control-group">
            <label>Loss Functions:</label>
            <div class="checkbox-group" id="loss_functions">
                <label><input type="checkbox" value="mse"> MSE</label>
                <label><input type="checkbox" value="mae"> MAE</label>
                <label><input type="checkbox" value="cross_entropy" checked> Cross Entropy</label>
                <label><input type="checkbox" value="sparse_categorical_crossentropy"> Sparse Cat CE</label>
                <label><input type="checkbox" value="bce"> BCE</label>
                <label><input type="checkbox" value="huber"> Huber</label>
            </div>
        </div>

        <button onclick="trainModels()" id="trainBtn">Train Models</button>
        <div id="loading"><h3>Training... please wait</h3></div>
        <div id="results"></div>
    </div>

    <!-- DOCUMENTATION TAB -->
    <div id="documentation-tab" class="tab-content">
        <h2 class="section-title">Loss Function Formulas</h2>

        <div class="loss-function-card">
            <h5>Mean Squared Error (MSE)</h5>
            <div class="formula">
                $$ L_{\\text{MSE}} = \\frac{1}{n} \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2 $$
            </div>
            <p>Best for regression. Sensitive to outliers.</p>
        </div>

        <div class="loss-function-card">
            <h5>Mean Absolute Error (MAE)</h5>
            <div class="formula">
                $$ L_{\\text{MAE}} = \\frac{1}{n} \\sum_{i=1}^{n} |y_i - \\hat{y}_i| $$
            </div>
            <p>Robust to outliers. Constant gradient.</p>
        </div>

        <div class="loss-function-card">
            <h5>Cross-Entropy Loss</h5>
            <div class="formula">
                $$ L_{\\text{CE}} = -\\sum_{c=1}^{C} y_c \\log(\\hat{p}_c) $$
            </div>
            <p>For multi-class classification with softmax.</p>
        </div>

        <div class="loss-function-card">
            <h5>Sparse Categorical Cross-Entropy</h5>
            <div class="formula">
                $$ L = -\\log(\\hat{p}_{y}) $$
            </div>
            <p>Same as CE, but label is integer index.</p>
        </div>

        <div class="loss-function-card">
            <h5>Binary Cross-Entropy (BCE)</h5>
            <div class="formula">
                $$ L = -[y \\log(\\sigma(z)) + (1-y) \\log(1-\\sigma(z))] $$
            </div>
            <p>For binary classification with sigmoid.</p>
        </div>

        <div class="loss-function-card">
            <h5>Huber Loss</h5>
            <div class="formula">
                $$ L_{\\delta} = \\begin{cases} 
                \\frac{1}{2}(y - \\hat{y})^2 & |y - \\hat{y}| \\leq \\delta \\\\
                \\delta(|y - \\hat{y}| - \\frac{1}{2}\\delta) & \\text{otherwise}
                \\end{cases} $$
            </div>
            <p>Combines MSE and MAE. Robust + smooth.</p>
        </div>
    </div>
</div>

<script>
function switchTab(tab) {
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-button').forEach(b => b.classList.remove('active'));
    document.getElementById(tab + '-tab').classList.add('active');
    event.target.classList.add('active');
    if (tab === 'documentation' && window.MathJax) {
        MathJax.typesetPromise();
    }
}

function updateLossOptions() {
    const task = document.getElementById('task_type').value;
    document.getElementById('n_classes_group').style.display = task === 'multi_classification' ? 'block' : 'none';
    const checks = document.querySelectorAll('#loss_functions input');
    checks.forEach(c => c.checked = false);
    if (task === 'regression') {
        document.querySelector('input[value="mse"]').checked = true;
    } else {
        document.querySelector('input[value="cross_entropy"]').checked = true;
        if (task === 'binary_classification') {
            document.querySelector('input[value="bce"]').checked = true;
        }
    }
}

async function trainModels() {
    const config = {
        task_type: document.getElementById('task_type').value,
        n_samples: parseInt(document.getElementById('n_samples').value),
        n_features: parseInt(document.getElementById('n_features').value),
        noise_level: parseFloat(document.getElementById('noise_level').value),
        outlier_percentage: parseFloat(document.getElementById('outlier_percentage').value),
        outlier_magnitude: parseFloat(document.getElementById('outlier_magnitude').value),
        n_hidden_layers: parseInt(document.getElementById('n_hidden_layers').value),
        neurons_per_layer: parseInt(document.getElementById('neurons_per_layer').value),
        epochs: parseInt(document.getElementById('epochs').value),
        loss_functions: Array.from(document.querySelectorAll('#loss_functions input:checked')).map(c => c.value)
    };
    if (config.task_type === 'multi_classification') {
        config.n_classes = parseInt(document.getElementById('n_classes').value);
    }
    if (config.loss_functions.length === 0) { alert('Select at least one loss'); return; }

    document.getElementById('trainBtn').disabled = true;
    document.getElementById('loading').style.display = 'block';
    document.getElementById('results').innerHTML = '';

    try {
        const res = await fetch('/train', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        displayResults(data, config.epochs, config.task_type);
    } catch (e) {
        document.getElementById('results').innerHTML = `<div style="background:#fee; color:#c66; padding:20px; border-radius:8px;">Error: ${e.message}</div>`;
    } finally {
        document.getElementById('trainBtn').disabled = false;
        document.getElementById('loading').style.display = 'none';
    }
}

function displayResults(data, epochs, task) {
    let html = '<h2>Loss Curves</h2>';
    Object.keys(data).forEach(name => {
        html += `<div class="plot-container"><h3>${name.toUpperCase()}</h3><div id="plot_${name}"></div></div>`;
    });
    html += `<div class="plot-container"><h3>Training Time</h3><div id="timePlot"></div></div>`;
    html += `<h2>Metrics</h2><table class="summary-table" id="metricsTable"></table>`;
    document.getElementById('results').innerHTML = html;

    Object.keys(data).forEach(name => {
        const d = data[name];
        Plotly.newPlot(`plot_${name}`, [
            { y: d.train_losses, x: [...Array(epochs).keys()].map(i => i+1), name: 'Train', line: {width: 2} },
            { y: d.val_losses, x: [...Array(epochs).keys()].map(i => i+1), name: 'Val', line: {width: 2, dash: 'dot'} }
        ], { yaxis: { type: 'log' }, height: 400 });
    });

    const times = Object.keys(data).map(k => data[k].total_time);
    Plotly.newPlot('timePlot', [{ x: Object.keys(data), y: times, type: 'bar' }], { height: 400 });

    let table = '<tr><th>Loss</th><th>Converged</th><th>Time (s)</th>';
    if (task === 'regression') table += '<th>MSE</th><th>MAE</th><th>R²</th></tr>';
    else table += '<th>Acc</th><th>Prec</th><th>Rec</th><th>F1</th></tr>';

    Object.keys(data).forEach(name => {
        const m = data[name].metrics;
        const conv = data[name].convergence_epoch;
        table += `<tr><td><strong>${name}</strong></td><td>${conv}</td><td>${data[name].total_time.toFixed(2)}</td>`;
        if (task === 'regression') {
            table += `<td>${m.mse.toFixed(4)}</td><td>${m.mae.toFixed(4)}</td><td>${m.r2.toFixed(4)}</td>`;
        } else {
            table += `<td>${m.accuracy.toFixed(4)}</td><td>${m.precision.toFixed(4)}</td><td>${m.recall.toFixed(4)}</td><td>${m.f1.toFixed(4)}</td>`;
        }
        table += '</tr>';
    });
    document.getElementById('metricsTable').innerHTML = table;
}
</script>
</body>
</html>'''

class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim, task_type, loss_name, n_hidden_layers, neurons_per_layer):
        super().__init__()
        self.task_type = task_type
        self.loss_name = loss_name
        layers = [nn.Linear(input_dim, neurons_per_layer), nn.ReLU()]
        for _ in range(n_hidden_layers - 1):
            layers.extend([nn.Linear(neurons_per_layer, neurons_per_layer), nn.ReLU()])
        layers.append(nn.Linear(neurons_per_layer, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def get_loss_function(name):
    return {
        'mse': nn.MSELoss(),
        'mae': nn.L1Loss(),
        'cross_entropy': nn.CrossEntropyLoss(),
        'sparse_categorical_crossentropy': nn.CrossEntropyLoss(),
        'bce': nn.BCEWithLogitsLoss(),
        'huber': nn.SmoothL1Loss()
    }.get(name, nn.MSELoss())

def evaluate_regression(model, X, y):
    model.eval()
    with torch.no_grad():
        pred = model(X).cpu().numpy()
        true = y.cpu().numpy()
    return {
        'mse': mean_squared_error(true, pred),
        'mae': mean_absolute_error(true, pred),
        'r2': r2_score(true, pred)
    }

def evaluate_classification(model, X, y):
    model.eval()
    with torch.no_grad():
        out = model(X)
        if model.loss_name == 'bce' and model.task_type == 'binary_classification':
            pred = (torch.sigmoid(out) > 0.5).long()
        else:
            _, pred = torch.max(out, 1)
        true = y.cpu().numpy()
        pred = pred.cpu().numpy()
    return {
        'accuracy': accuracy_score(true, pred),
        'precision': precision_score(true, pred, average='macro', zero_division=0),
        'recall': recall_score(true, pred, average='macro', zero_division=0),
        'f1': f1_score(true, pred, average='macro', zero_division=0)
    }

def add_outliers(X, y, n_samples, perc, mag):
    n = int(n_samples * perc / 100)
    if n == 0: return X, y
    idx = np.random.choice(n_samples, n, replace=False)
    X[idx] *= mag
    if y.ndim == 2:  # regression
        y[idx] *= mag
    else:  # classification
        y[idx] = 1 - y[idx]
    return X, y

def clean_nans(d):
    if isinstance(d, list):
        return [None if np.isnan(x) else x for x in d]
    if isinstance(d, dict):
        return {k: clean_nans(v) for k, v in d.items()}
    return d

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'}), 200

@app.route('/train', methods=['POST'])
def train():
    try:
        cfg = request.get_json()
        if not cfg: return jsonify({'error': 'No data'}), 400

        task = cfg['task_type']
        n_samples = cfg['n_samples']
        n_features = cfg['n_features']
        noise = cfg['noise_level']
        out_p = cfg['outlier_percentage']
        out_m = cfg['outlier_magnitude']
        layers = cfg['n_hidden_layers']
        neurons = cfg['neurons_per_layer']
        epochs = cfg['epochs']
        losses = cfg['loss_functions']
        n_classes = cfg.get('n_classes')

        # Generate data
        if task == 'regression':
            X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise, random_state=42)
            y = y.reshape(-1, 1)
            output_dim = 1
        else:
            flip = min(1.0, noise / 100)
            n_cls = 2 if task == 'binary_classification' else n_classes
            X, y = make_classification(n_samples=n_samples, n_features=n_features, n_classes=n_cls,
                                       n_informative=max(2, n_features//2), n_clusters_per_class=1,
                                       flip_y=flip, random_state=42)
            output_dim = n_cls

        X, y = add_outliers(X, y, n_samples, out_p, out_m)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        X_train_t = torch.FloatTensor(X_train)
        X_val_t = torch.FloatTensor(X_val)
        y_train_t = torch.FloatTensor(y_train) if task == 'regression' else torch.LongTensor(y_train)
        y_val_t = torch.FloatTensor(y_val) if task == 'regression' else torch.LongTensor(y_val)

        results = {}
        for loss_name in losses:
            out_dim = 1 if task == 'binary_classification' and loss_name == 'bce' else output_dim
            model = SimpleNN(n_features, out_dim, task, loss_name, layers, neurons)
            criterion = get_loss_function(loss_name)
            opt = optim.Adam(model.parameters(), lr=0.001)

            train_losses, val_losses = [], []
            start = time.time()
            conv_epoch = None

            for epoch in range(epochs):
                model.train()
                opt.zero_grad()
                out = model(X_train_t)

                try:
                    if task != 'regression' and loss_name in ['cross_entropy', 'sparse_categorical_crossentropy']:
                        loss = criterion(out, y_train_t)
                    elif task == 'binary_classification' and loss_name == 'bce':
                        loss = criterion(out, y_train_t.float())
                    else:
                        loss = criterion(out, y_train_t.float() if task != 'regression' else y_train_t)
                except:
                    loss = torch.tensor(0.0)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                model.eval()
                with torch.no_grad():
                    val_out = model(X_val_t)
                    try:
                        val_loss = criterion(val_out, y_val_t.float() if task != 'regression' and loss_name != 'bce' else y_val_t)
                    except:
                        val_loss = torch.tensor(0.0)

                train_losses.append(float(loss.item()))
                val_losses.append(float(val_loss.item()))

                if epoch > 10 and conv_epoch is None and len(val_losses) >= 5:
                    recent = val_losses[-5:]
                    if abs(recent[-1] - recent[0]) / (abs(recent[0]) + 1e-8) < 0.001:
                        conv_epoch = epoch

            total_time = time.time() - start
            metrics = evaluate_regression(model, X_val_t, y_val_t) if task == 'regression' else evaluate_classification(model, X_val_t, y_val_t)

            results[loss_name] = {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'total_time': total_time,
                'convergence_epoch': conv_epoch or epochs,
                'metrics': metrics
            }

        return jsonify(clean_nans(results))

    except Exception as e:
        logger.error(f"Training error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False, threaded=True)