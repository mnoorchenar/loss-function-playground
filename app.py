from flask import Flask, render_template_string, jsonify, request
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import threading
import time
import os

app = Flask(__name__)

# Global variables for training state
training_state = {
    'is_training': False,
    'progress': 0,
    'current_epoch': 0,
    'total_epochs': 0,
    'losses': [],
    'trained': False,
    'current_loss': 0
}

# VAE Architecture
class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=2):
        super(VAE, self).__init__()
        
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
        
        self.latent_dim = latent_dim
        
    def encode(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Loss function
def vae_loss(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD, BCE, KLD

# Load MNIST data
def load_mnist_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    
    # Get subset for faster training and visualization
    subset_size = 10000
    indices = torch.randperm(len(train_dataset))[:subset_size]
    
    data = []
    labels = []
    
    for idx in indices:
        img, label = train_dataset[idx]
        data.append(img.view(-1).numpy())
        labels.append(label)
    
    return np.array(data), np.array(labels)

# Initialize model and data
print("Loading MNIST dataset...")
vae = None
data, labels = load_mnist_data()
data_tensor = torch.FloatTensor(data)
print(f"Loaded {len(data)} MNIST samples")

# Train the VAE in a separate thread
def train_vae_thread(epochs, batch_size, learning_rate, hidden_dim, latent_dim):
    global vae, training_state
    
    training_state['is_training'] = True
    training_state['progress'] = 0
    training_state['current_epoch'] = 0
    training_state['total_epochs'] = epochs
    training_state['losses'] = []
    
    # Initialize new model with specified parameters
    vae = VAE(input_dim=784, hidden_dim=hidden_dim, latent_dim=latent_dim)
    optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)
    dataset = torch.utils.data.TensorDataset(data_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        vae.train()
        total_loss = 0
        batch_count = 0
        
        for batch in dataloader:
            x = batch[0]
            optimizer.zero_grad()
            recon_x, mu, logvar = vae(x)
            loss, _, _ = vae_loss(recon_x, x, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1
        
        avg_loss = total_loss / len(dataloader.dataset)
        training_state['losses'].append(avg_loss)
        training_state['current_epoch'] = epoch + 1
        training_state['current_loss'] = avg_loss
        training_state['progress'] = int(((epoch + 1) / epochs) * 100)
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    training_state['is_training'] = False
    training_state['trained'] = True
    print("Training complete!")

def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return img_str

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>VAE Interactive Playground</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        h1 {
            text-align: center;
            color: #667eea;
            margin-bottom: 10px;
            font-size: 2.5em;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
            font-size: 1.1em;
        }
        .tab-container {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            border-bottom: 2px solid #eee;
            flex-wrap: wrap;
        }
        .tab {
            padding: 12px 24px;
            background: none;
            border: none;
            cursor: pointer;
            font-size: 16px;
            color: #666;
            border-bottom: 3px solid transparent;
            transition: all 0.3s;
        }
        .tab:hover {
            color: #667eea;
        }
        .tab.active {
            color: #667eea;
            border-bottom-color: #667eea;
            font-weight: 600;
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .card {
            background: #f8f9fa;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .card h3 {
            color: #333;
            margin-bottom: 15px;
            font-size: 1.3em;
        }
        .card img {
            width: 100%;
            border-radius: 8px;
            margin-top: 10px;
        }
        .slider-container {
            margin: 15px 0;
        }
        .slider-container label {
            display: block;
            margin-bottom: 8px;
            color: #555;
            font-weight: 500;
        }
        .slider {
            width: 100%;
            height: 8px;
            border-radius: 5px;
            background: #ddd;
            outline: none;
        }
        .slider::-webkit-slider-thumb {
            appearance: none;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: #667eea;
            cursor: pointer;
        }
        .value-display {
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.9em;
            margin-left: 10px;
        }
        button {
            background: #667eea;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 16px;
            transition: all 0.3s;
            margin: 10px 5px;
        }
        button:hover {
            background: #5568d3;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }
        button:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }
        .architecture-box {
            background: white;
            border: 2px solid #667eea;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
            text-align: center;
        }
        .arrow {
            text-align: center;
            font-size: 24px;
            color: #667eea;
            margin: 5px 0;
        }
        .info-box {
            background: #e3f2fd;
            border-left: 4px solid #2196F3;
            padding: 15px;
            margin: 15px 0;
            border-radius: 4px;
        }
        .loading {
            text-align: center;
            padding: 20px;
            color: #666;
        }
        .training-controls {
            background: #fff;
            border: 2px solid #667eea;
            border-radius: 12px;
            padding: 25px;
            margin: 20px 0;
        }
        .input-group {
            margin: 15px 0;
        }
        .input-group label {
            display: block;
            margin-bottom: 5px;
            color: #555;
            font-weight: 500;
        }
        .input-group input, .input-group select {
            width: 100%;
            padding: 10px;
            border: 2px solid #ddd;
            border-radius: 6px;
            font-size: 14px;
        }
        .input-group input:focus {
            outline: none;
            border-color: #667eea;
        }
        .progress-container {
            background: #f0f0f0;
            border-radius: 10px;
            height: 30px;
            margin: 20px 0;
            overflow: hidden;
            position: relative;
        }
        .progress-bar {
            background: linear-gradient(90deg, #667eea, #764ba2);
            height: 100%;
            transition: width 0.3s;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }
        .status-badge {
            display: inline-block;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: 600;
            margin: 10px 5px;
        }
        .status-training {
            background: #ffc107;
            color: #000;
        }
        .status-ready {
            background: #4caf50;
            color: white;
        }
        .status-not-trained {
            background: #f44336;
            color: white;
        }
        .training-info {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
        }
        .training-info p {
            margin: 5px 0;
            color: #555;
        }
        .param-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 Variational Autoencoder Playground</h1>
        <p class="subtitle">Interactive visualization for understanding VAE architecture and latent space</p>
        
        <div class="tab-container">
            <button class="tab active" onclick="switchTab('training')">Training Dashboard</button>
            <button class="tab" onclick="switchTab('architecture')">Architecture</button>
            <button class="tab" onclick="switchTab('latent')">Latent Space</button>
            <button class="tab" onclick="switchTab('reconstruction')">Reconstruction</button>
            <button class="tab" onclick="switchTab('generation')">Generation</button>
        </div>
        
        <div id="training" class="tab-content active">
            <div class="training-controls">
                <h3>⚙️ Training Configuration</h3>
                <p style="color: #666; margin-bottom: 20px;">Configure your VAE parameters and start training</p>
                
                <div class="param-grid">
                    <div class="input-group">
                        <label>Number of Epochs</label>
                        <input type="number" id="epochs" value="30" min="1" max="200">
                    </div>
                    
                    <div class="input-group">
                        <label>Batch Size</label>
                        <select id="batch_size">
                            <option value="32">32</option>
                            <option value="64">64</option>
                            <option value="128" selected>128</option>
                            <option value="256">256</option>
                        </select>
                    </div>
                    
                    <div class="input-group">
                        <label>Learning Rate</label>
                        <select id="learning_rate">
                            <option value="0.0001">0.0001</option>
                            <option value="0.001" selected>0.001</option>
                            <option value="0.01">0.01</option>
                        </select>
                    </div>
                    
                    <div class="input-group">
                        <label>Hidden Dimension</label>
                        <select id="hidden_dim">
                            <option value="200">200</option>
                            <option value="400" selected>400</option>
                            <option value="512">512</option>
                        </select>
                    </div>
                    
                    <div class="input-group">
                        <label>Latent Dimension</label>
                        <select id="latent_dim">
                            <option value="2" selected>2</option>
                            <option value="5">5</option>
                            <option value="10">10</option>
                            <option value="20">20</option>
                        </select>
                    </div>
                </div>
                
                <div style="margin-top: 20px;">
                    <button id="train-btn" onclick="startTraining()">🚀 Start Training</button>
                    <button onclick="resetModel()">🔄 Reset Model</button>
                </div>
            </div>
            
            <div class="training-info">
                <h3>📊 Training Status</h3>
                <p><strong>Status:</strong> <span id="status-badge" class="status-badge status-not-trained">Not Trained</span></p>
                <p id="epoch-info"><strong>Epoch:</strong> 0 / 0</p>
                <p id="loss-info"><strong>Current Loss:</strong> N/A</p>
            </div>
            
            <div id="progress-section" style="display: none;">
                <h3>Training Progress</h3>
                <div class="progress-container">
                    <div class="progress-bar" id="progress-bar" style="width: 0%">0%</div>
                </div>
            </div>
            
            <div class="card" id="loss-curve-card" style="display: none;">
                <h3>Real-time Training Loss</h3>
                <div id="training-plot"></div>
                <button onclick="updateLossCurve()">Refresh Loss Curve</button>
            </div>
        </div>
        
        <div id="architecture" class="tab-content">
            <div class="info-box">
                <strong>VAE Architecture:</strong> A Variational Autoencoder learns to compress data into a lower-dimensional latent space and reconstruct it. 
                The key innovation is the reparameterization trick, which allows backpropagation through stochastic sampling.
            </div>
            
            <div class="architecture-box">
                <h4>Input (784D)</h4>
                <small>28×28 image flattened</small>
            </div>
            <div class="arrow">↓</div>
            <div class="architecture-box" style="background: #fff3e0;">
                <h4>Encoder: FC Layer (<span id="arch-hidden">400</span>D)</h4>
                <small>ReLU activation</small>
            </div>
            <div class="arrow">↓</div>
            <div class="architecture-box" style="background: #e8f5e9;">
                <h4>Latent Space (<span id="arch-latent">2</span>D)</h4>
                <small>μ (mean) and σ² (variance)</small>
            </div>
            <div class="arrow">↓ Reparameterization Trick</div>
            <div class="architecture-box" style="background: #e8f5e9;">
                <h4>Sample z ~ N(μ, σ²)</h4>
                <small>z = μ + σ * ε, where ε ~ N(0,1)</small>
            </div>
            <div class="arrow">↓</div>
            <div class="architecture-box" style="background: #f3e5f5;">
                <h4>Decoder: FC Layer (<span id="arch-hidden2">400</span>D)</h4>
                <small>ReLU activation</small>
            </div>
            <div class="arrow">↓</div>
            <div class="architecture-box">
                <h4>Output (784D)</h4>
                <small>Reconstructed image</small>
            </div>
            
            <div class="info-box" style="background: #fff3e0; border-left-color: #ff9800; margin-top: 20px;">
                <strong>Loss Function:</strong> VAE Loss = Reconstruction Loss (BCE) + KL Divergence<br>
                • BCE: Measures how well we reconstruct the input<br>
                • KLD: Regularizes latent space to be close to N(0,1)
            </div>
        </div>
        
        <div id="latent" class="tab-content">
            <div class="info-box" style="background: #fff3e0; border-left-color: #ff9800;">
                ⚠️ Please train the model first in the Training Dashboard before using this feature.
            </div>
            <div class="card">
                <h3>Latent Space Visualization</h3>
                <p>Each point represents an MNIST digit encoded in 2D latent space. Colors indicate digit classes (0-9).</p>
                <button onclick="loadLatentSpace()">Refresh Latent Space</button>
                <div id="latent-plot" class="loading">Train the model first, then click button to generate...</div>
            </div>
        </div>
        
        <div id="reconstruction" class="tab-content">
            <div class="info-box" style="background: #fff3e0; border-left-color: #ff9800;">
                ⚠️ Please train the model first in the Training Dashboard before using this feature.
            </div>
            <div class="card">
                <h3>Input vs Reconstruction</h3>
                <p>See how well the VAE reconstructs MNIST digits.</p>
                <button onclick="loadReconstruction()">Show Random Reconstruction</button>
                <div id="recon-plot" class="loading">Train the model first, then click button to generate...</div>
            </div>
        </div>
        
        <div id="generation" class="tab-content">
            <div class="info-box" style="background: #fff3e0; border-left-color: #ff9800;">
                ⚠️ Please train the model first in the Training Dashboard before using this feature. Generation works best with 2D latent space.
            </div>
            <div class="card">
                <h3>Generate from Latent Space</h3>
                <p>Manipulate latent dimensions to generate new digit-like samples. Explore how different regions of latent space correspond to different digits!</p>
                
                <div class="slider-container">
                    <label>Z1 (Latent Dimension 1): <span class="value-display" id="z1-val">0.00</span></label>
                    <input type="range" class="slider" id="z1" min="-3" max="3" step="0.1" value="0" oninput="updateValue('z1')">
                </div>
                
                <div class="slider-container">
                    <label>Z2 (Latent Dimension 2): <span class="value-display" id="z2-val">0.00</span></label>
                    <input type="range" class="slider" id="z2" min="-3" max="3" step="0.1" value="0" oninput="updateValue('z2')">
                </div>
                
                <button onclick="generateSample()">Generate Image</button>
                <button onclick="randomSample()">Random Sample</button>
                <button onclick="generateGrid()">Generate Grid (2D only)</button>
                
                <div id="gen-plot" class="loading">Train the model first, then adjust sliders and click Generate...</div>
            </div>
        </div>
    </div>
    
    <script>
        let progressInterval = null;
        
        function switchTab(tabName) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            event.target.classList.add('active');
            document.getElementById(tabName).classList.add('active');
        }
        
        function updateValue(id) {
            const val = document.getElementById(id).value;
            document.getElementById(id + '-val').textContent = parseFloat(val).toFixed(2);
        }
        
        async function startTraining() {
            const epochs = parseInt(document.getElementById('epochs').value);
            const batch_size = parseInt(document.getElementById('batch_size').value);
            const learning_rate = parseFloat(document.getElementById('learning_rate').value);
            const hidden_dim = parseInt(document.getElementById('hidden_dim').value);
            const latent_dim = parseInt(document.getElementById('latent_dim').value);
            
            // Update architecture display
            document.getElementById('arch-hidden').textContent = hidden_dim;
            document.getElementById('arch-hidden2').textContent = hidden_dim;
            document.getElementById('arch-latent').textContent = latent_dim;
            
            document.getElementById('train-btn').disabled = true;
            document.getElementById('progress-section').style.display = 'block';
            document.getElementById('loss-curve-card').style.display = 'block';
            
            const response = await fetch('/start_training', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({epochs, batch_size, learning_rate, hidden_dim, latent_dim})
            });
            
            const data = await response.json();
            
            if (data.status === 'started') {
                // Start polling for progress
                progressInterval = setInterval(updateProgress, 500);
            }
        }
        
        async function updateProgress() {
            const response = await fetch('/training_progress');
            const data = await response.json();
            
            const progressBar = document.getElementById('progress-bar');
            progressBar.style.width = data.progress + '%';
            progressBar.textContent = data.progress + '%';
            
            document.getElementById('epoch-info').innerHTML = `<strong>Epoch:</strong> ${data.current_epoch} / ${data.total_epochs}`;
            document.getElementById('loss-info').innerHTML = `<strong>Current Loss:</strong> ${data.current_loss.toFixed(4)}`;
            
            const statusBadge = document.getElementById('status-badge');
            if (data.is_training) {
                statusBadge.className = 'status-badge status-training';
                statusBadge.textContent = 'Training...';
            } else if (data.trained) {
                statusBadge.className = 'status-badge status-ready';
                statusBadge.textContent = 'Ready';
                document.getElementById('train-btn').disabled = false;
                clearInterval(progressInterval);
                updateLossCurve();
            } else {
                statusBadge.className = 'status-badge status-not-trained';
                statusBadge.textContent = 'Not Trained';
            }
        }
        
        async function updateLossCurve() {
            const response = await fetch('/training_curve');
            const data = await response.json();
            if (data.image) {
                document.getElementById('training-plot').innerHTML = `<img src="data:image/png;base64,${data.image}">`;
            }
        }
        
        async function resetModel() {
            if (confirm('Are you sure you want to reset the model? All training progress will be lost.')) {
                const response = await fetch('/reset_model', {method: 'POST'});
                const data = await response.json();
                if (data.status === 'reset') {
                    location.reload();
                }
            }
        }
        
        async function loadLatentSpace() {
            document.getElementById('latent-plot').innerHTML = '<div class="loading">Generating...</div>';
            const response = await fetch('/latent_space');
            const data = await response.json();
            if (data.error) {
                document.getElementById('latent-plot').innerHTML = `<div class="loading" style="color: red;">${data.error}</div>`;
            } else {
                document.getElementById('latent-plot').innerHTML = `<img src="data:image/png;base64,${data.image}">`;
            }
        }
        
        async function loadReconstruction() {
            document.getElementById('recon-plot').innerHTML = '<div class="loading">Generating...</div>';
            const response = await fetch('/reconstruction');
            const data = await response.json();
            if (data.error) {
                document.getElementById('recon-plot').innerHTML = `<div class="loading" style="color: red;">${data.error}</div>`;
            } else {
                document.getElementById('recon-plot').innerHTML = `<img src="data:image/png;base64,${data.image}">`;
            }
        }
        
        async function generateSample() {
            const z1 = parseFloat(document.getElementById('z1').value);
            const z2 = parseFloat(document.getElementById('z2').value);
            document.getElementById('gen-plot').innerHTML = '<div class="loading">Generating...</div>';
            const response = await fetch('/generate', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({z1, z2})
            });
            const data = await response.json();
            if (data.error) {
                document.getElementById('gen-plot').innerHTML = `<div class="loading" style="color: red;">${data.error}</div>`;
            } else {
                document.getElementById('gen-plot').innerHTML = `<img src="data:image/png;base64,${data.image}">`;
            }
        }
        
        async function randomSample() {
            const z1 = (Math.random() * 6 - 3).toFixed(2);
            const z2 = (Math.random() * 6 - 3).toFixed(2);
            document.getElementById('z1').value = z1;
            document.getElementById('z2').value = z2;
            updateValue('z1');
            updateValue('z2');
            await generateSample();
        }
        
        async function generateGrid() {
            document.getElementById('gen-plot').innerHTML = '<div class="loading">Generating grid...</div>';
            const response = await fetch('/generate_grid');
            const data = await response.json();
            if (data.error) {
                document.getElementById('gen-plot').innerHTML = `<div class="loading" style="color: red;">${data.error}</div>`;
            } else {
                document.getElementById('gen-plot').innerHTML = `<img src="data:image/png;base64,${data.image}">`;
            }
        }
        
        // Check initial status
        updateProgress();
    </script>

    <div style="text-align: center; padding: 20px; color: #666; margin-top: 40px; border-top: 2px solid #eee;">
    <p>Created by <strong>Mohammad Noorchenarboo</strong> | 
    <a href="https://www.linkedin.com/in/mnoorchenar" target="_blank" style="color: #667eea;">LinkedIn</a></p>
</div>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/start_training', methods=['POST'])
def start_training():
    global training_state
    
    if training_state['is_training']:
        return jsonify({'status': 'already_training'})
    
    params = request.json
    epochs = params.get('epochs', 30)
    batch_size = params.get('batch_size', 128)
    learning_rate = params.get('learning_rate', 0.001)
    hidden_dim = params.get('hidden_dim', 400)
    latent_dim = params.get('latent_dim', 2)
    
    # Start training in a separate thread
    thread = threading.Thread(
        target=train_vae_thread,
        args=(epochs, batch_size, learning_rate, hidden_dim, latent_dim)
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({'status': 'started'})

@app.route('/training_progress')
def training_progress():
    return jsonify({
        'is_training': training_state['is_training'],
        'progress': training_state['progress'],
        'current_epoch': training_state['current_epoch'],
        'total_epochs': training_state['total_epochs'],
        'current_loss': training_state['current_loss'],
        'trained': training_state['trained']
    })

@app.route('/reset_model', methods=['POST'])
def reset_model():
    global vae, training_state
    vae = None
    training_state = {
        'is_training': False,
        'progress': 0,
        'current_epoch': 0,
        'total_epochs': 0,
        'losses': [],
        'trained': False,
        'current_loss': 0
    }
    return jsonify({'status': 'reset'})

@app.route('/latent_space')
def latent_space():
    if vae is None or not training_state['trained']:
        return jsonify({'error': 'Model not trained yet. Please train the model first.'})
    
    if vae.latent_dim != 2:
        return jsonify({'error': 'Latent space visualization only works with 2D latent dimension.'})
    
    vae.eval()
    with torch.no_grad():
        mu, _ = vae.encode(data_tensor)
        mu_np = mu.numpy()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    scatter = ax.scatter(mu_np[:, 0], mu_np[:, 1], c=labels, cmap='tab10', 
                         alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Latent Dimension 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latent Dimension 2', fontsize=12, fontweight='bold')
    ax.set_title('VAE Latent Space - MNIST Digits (2D)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax, ticks=range(10))
    cbar.set_label('Digit Class', fontsize=11)
    cbar.ax.set_yticklabels(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    
    return jsonify({'image': fig_to_base64(fig)})

@app.route('/reconstruction')
def reconstruction():
    if vae is None or not training_state['trained']:
        return jsonify({'error': 'Model not trained yet. Please train the model first.'})
    
    # Show multiple reconstructions
    n_samples = 10
    indices = np.random.choice(len(data), n_samples, replace=False)
    
    vae.eval()
    with torch.no_grad():
        originals = data_tensor[indices]
        reconstructions, _, _ = vae(originals)
    
    fig, axes = plt.subplots(2, n_samples, figsize=(20, 4))
    
    for i in range(n_samples):
        # Original
        axes[0, i].imshow(originals[i].numpy().reshape(28, 28), cmap='gray')
        axes[0, i].set_title(f'Original\n(Digit {labels[indices[i]]})', fontsize=9)
        axes[0, i].axis('off')
        
        # Reconstruction
        axes[1, i].imshow(reconstructions[i].numpy().reshape(28, 28), cmap='gray')
        axes[1, i].set_title('Reconstructed', fontsize=9)
        axes[1, i].axis('off')
    
    fig.suptitle('MNIST Reconstruction Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return jsonify({'image': fig_to_base64(fig)})

@app.route('/generate', methods=['POST'])
def generate():
    if vae is None or not training_state['trained']:
        return jsonify({'error': 'Model not trained yet. Please train the model first.'})
    
    data = request.json
    z1 = data['z1']
    z2 = data['z2']
    
    # Create latent vector with correct dimensions
    if vae.latent_dim == 2:
        z = torch.FloatTensor([[z1, z2]])
    else:
        # For higher dimensions, use z1 and z2 for first two dims, zeros for rest
        z = torch.zeros(1, vae.latent_dim)
        z[0, 0] = z1
        z[0, 1] = z2
    
    vae.eval()
    with torch.no_grad():
        generated = vae.decode(z)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(generated.numpy().reshape(28, 28), cmap='gray')
    ax.set_title(f'Generated Digit\nz1={z1:.2f}, z2={z2:.2f}', 
                 fontsize=13, fontweight='bold')
    ax.axis('off')
    
    return jsonify({'image': fig_to_base64(fig)})

@app.route('/generate_grid')
def generate_grid():
    if vae is None or not training_state['trained']:
        return jsonify({'error': 'Model not trained yet. Please train the model first.'})
    
    if vae.latent_dim != 2:
        return jsonify({'error': 'Grid generation only works with 2D latent dimension.'})
    
    # Generate a grid of images by sampling latent space
    n = 15
    grid_x = np.linspace(-3, 3, n)
    grid_y = np.linspace(-3, 3, n)
    
    fig, axes = plt.subplots(n, n, figsize=(15, 15))
    
    vae.eval()
    with torch.no_grad():
        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z = torch.FloatTensor([[xi, yi]])
                generated = vae.decode(z)
                axes[i, j].imshow(generated.numpy().reshape(28, 28), cmap='gray')
                axes[i, j].axis('off')
    
    fig.suptitle('Latent Space Manifold (15×15 Grid)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return jsonify({'image': fig_to_base64(fig)})

@app.route('/training_curve')
def training_curve():
    if not training_state['losses']:
        return jsonify({'error': 'No training data available yet.'})
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(training_state['losses'], linewidth=2, color='#667eea')
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title('VAE Training Loss Over Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.fill_between(range(len(training_state['losses'])), training_state['losses'], alpha=0.3, color='#667eea')
    
    return jsonify({'image': fig_to_base64(fig)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)