import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import scipy.fft as fft
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Config
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import warnings
warnings.filterwarnings('ignore')

class SolarWindData:
    def __init__(self, start_date='1981-01-01', duration_years=1):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = self.start_date + timedelta(days=365 * duration_years)
        self.time_series = pd.date_range(start=self.start_date, end=self.end_date, freq='1T')
        self.n_samples = len(self.time_series)
        self.data = self._generate_synthetic_data()
    
    def _generate_synthetic_data(self):
        """Generate synthetic solar wind data using multiple cosine waves + noise"""
        t = np.arange(self.n_samples) / (60 * 24)  # Convert to days
        
        # Base frequencies (cycles per day)
        frequencies = [0.5, 1.0, 2.3, 5.7, 11.2]  # Different periodicities
        
        data = {}
        
        # Generate B vector (magnetic field) components
        for axis in ['Bx', 'By', 'Bz']:
            signal = np.zeros(self.n_samples)
            amplitudes = np.random.uniform(5, 15, len(frequencies))  # Different amplitudes
            phases = np.random.uniform(0, 2*np.pi, len(frequencies))  # Different phases
            
            for freq, amp, phase in zip(frequencies, amplitudes, phases):
                signal += amp * np.cos(2 * np.pi * freq * t + phase)
            
            # Add Gaussian noise
            noise = np.random.normal(0, 2, self.n_samples)
            data[axis] = signal + noise
        
        # Generate V vector (velocity) components
        for axis in ['Vx', 'Vy', 'Vz']:
            signal = np.zeros(self.n_samples)
            amplitudes = np.random.uniform(300, 600, len(frequencies))  # Higher values for velocity
            phases = np.random.uniform(0, 2*np.pi, len(frequencies))
            
            for freq, amp, phase in zip(frequencies, amplitudes, phases):
                signal += amp * np.cos(2 * np.pi * freq * t + phase)
            
            # Add Gaussian noise
            noise = np.random.normal(0, 50, self.n_samples)
            data[axis] = signal + noise
        
        df = pd.DataFrame(data, index=self.time_series)
        return df
    
    def create_gaps(self, gap_probability=0.01, max_gap_minutes=60):
        """Create random gaps in the data"""
        data_with_gaps = self.data.copy()
        
        # Create random gaps
        np.random.seed(42)  # For reproducibility
        gap_starts = np.random.random(self.n_samples) < gap_probability
        
        for i, is_gap_start in enumerate(gap_starts):
            if is_gap_start:
                gap_duration = np.random.randint(5, max_gap_minutes + 1)
                end_idx = min(i + gap_duration, self.n_samples)
                data_with_gaps.iloc[i:end_idx] = np.nan
        
        return data_with_gaps

class GapFillingMethods:
    @staticmethod
    def interpolation_fill(data, method='linear'):
        """Simple interpolation method"""
        return data.interpolate(method=method)
    
    @staticmethod
    def fft_fill(data, column, gap_mask):
        """Improved FFT-based gap filling with proper frequency scaling and continuity"""
        filled_data = data[column].copy()
        
        # Find gap regions
        gap_indices = np.where(gap_mask)[0]
        if len(gap_indices) == 0:
            return filled_data
        
        # Group consecutive gaps
        gap_groups = []
        current_group = [gap_indices[0]]
        
        for i in range(1, len(gap_indices)):
            if gap_indices[i] - gap_indices[i-1] == 1:
                current_group.append(gap_indices[i])
            else:
                gap_groups.append(current_group)
                current_group = [gap_indices[i]]
        gap_groups.append(current_group)
        
        # Fill each gap group
        for gap_group in gap_groups:
            start_gap, end_gap = gap_group[0], gap_group[-1]
            gap_length = len(gap_group)
            
            # Adaptive context size based on gap length
            context_size = max(60, gap_length * 3)  # At least 3x gap length or 1 hour
            
            # Get left context (before gap)
            left_start = max(0, start_gap - context_size)
            left_context = filled_data[left_start:start_gap].dropna()
            
            # Get right context (after gap)
            right_end = min(len(data), end_gap + context_size + 1)
            right_context = filled_data[end_gap + 1:right_end].dropna()
            
            if len(left_context) >= 10 and len(right_context) >= 10:
                # Combine contexts for FFT analysis
                combined_context = pd.concat([left_context, right_context])
                
                if len(combined_context) > 20:
                    # Perform FFT on combined context
                    context_values = combined_context.values
                    fft_values = fft.fft(context_values)
                    frequencies = fft.fftfreq(len(context_values))
                    
                    # Filter out very high frequencies (noise) and DC component
                    magnitude = np.abs(fft_values)
                    valid_freqs = np.where((np.abs(frequencies) > 0.01) & (np.abs(frequencies) < 0.4))[0]
                    
                    if len(valid_freqs) > 0:
                        # Get top frequency components from valid range
                        valid_magnitudes = magnitude[valid_freqs]
                        top_indices = valid_freqs[np.argsort(valid_magnitudes)[-min(3, len(valid_freqs)):]]
                        
                        # Get boundary values for continuity
                        left_boundary = left_context.iloc[-1] if len(left_context) > 0 else 0
                        right_boundary = right_context.iloc[0] if len(right_context) > 0 else 0
                        
                        # Create time vector for gap (relative to left boundary)
                        t_gap = np.linspace(0, gap_length - 1, gap_length)
                        
                        # Reconstruct using selected frequencies with proper phase alignment
                        reconstructed = np.zeros(gap_length, dtype=complex)
                        total_weight = 0
                        
                        for freq_idx in top_indices:
                            freq = frequencies[freq_idx]
                            amplitude = fft_values[freq_idx]
                            weight = magnitude[freq_idx]
                            
                            # Scale amplitude to avoid spikes
                            scaled_amplitude = amplitude / len(context_values)
                            
                            # Add frequency component
                            reconstructed += weight * scaled_amplitude * np.exp(2j * np.pi * freq * t_gap)
                            total_weight += weight
                        
                        if total_weight > 0:
                            # Take real part and normalize
                            gap_values = np.real(reconstructed / total_weight)
                            
                            # Apply linear trend to ensure continuity
                            linear_trend = np.linspace(left_boundary, right_boundary, gap_length)
                            
                            # Blend FFT reconstruction with linear trend (weighted average)
                            # More linear trend for shorter gaps, more FFT for longer gaps
                            linear_weight = max(0.3, 1.0 - gap_length / 50.0)
                            fft_weight = 1.0 - linear_weight
                            
                            # Remove any remaining DC offset from FFT component
                            gap_values_centered = gap_values - np.mean(gap_values)
                            gap_mean = (left_boundary + right_boundary) / 2
                            gap_values_final = gap_values_centered + gap_mean
                            
                            final_values = (linear_weight * linear_trend + 
                                          fft_weight * gap_values_final)
                            
                            # Apply smoothing to reduce any remaining spikes
                            if gap_length > 2:
                                # Simple moving average smoothing
                                kernel_size = min(3, gap_length)
                                kernel = np.ones(kernel_size) / kernel_size
                                if gap_length >= kernel_size:
                                    final_values = np.convolve(final_values, kernel, mode='same')
                            
                            filled_data.iloc[gap_group] = final_values
                        else:
                            # Fallback to linear interpolation if FFT fails
                            linear_values = np.linspace(left_boundary, right_boundary, gap_length)
                            filled_data.iloc[gap_group] = linear_values
                    else:
                        # Fallback to linear interpolation if no valid frequencies
                        left_boundary = left_context.iloc[-1] if len(left_context) > 0 else 0
                        right_boundary = right_context.iloc[0] if len(right_context) > 0 else 0
                        linear_values = np.linspace(left_boundary, right_boundary, gap_length)
                        filled_data.iloc[gap_group] = linear_values
                else:
                    # Not enough context data, use linear interpolation
                    left_boundary = left_context.iloc[-1] if len(left_context) > 0 else 0
                    right_boundary = right_context.iloc[0] if len(right_context) > 0 else 0
                    linear_values = np.linspace(left_boundary, right_boundary, gap_length)
                    filled_data.iloc[gap_group] = linear_values
            else:
                # Insufficient context on one or both sides, use available data
                if len(left_context) > 0 and len(right_context) > 0:
                    left_val = left_context.iloc[-1]
                    right_val = right_context.iloc[0]
                    linear_values = np.linspace(left_val, right_val, gap_length)
                elif len(left_context) > 0:
                    # Forward fill from left context
                    linear_values = np.full(gap_length, left_context.iloc[-1])
                elif len(right_context) > 0:
                    # Backward fill from right context
                    linear_values = np.full(gap_length, right_context.iloc[0])
                else:
                    # No context available, use series mean
                    series_mean = filled_data.dropna().mean()
                    linear_values = np.full(gap_length, series_mean)
                
                filled_data.iloc[gap_group] = linear_values
        
        return filled_data
    
    @staticmethod
    def lstm_fill(data, column, gap_mask, sequence_length=60):
        """LSTM-based gap filling with CuDNN compatibility fixes"""
        filled_data = data[column].copy()
        
        # Prepare training data from non-gap regions
        non_gap_data = filled_data.dropna().values
        
        if len(non_gap_data) < sequence_length * 2:
            return filled_data  # Not enough data for LSTM
        
        # Normalize data
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(non_gap_data.reshape(-1, 1)).flatten()
        
        # Create sequences
        X, y = [], []
        for i in range(len(normalized_data) - sequence_length):
            X.append(normalized_data[i:i + sequence_length])
            y.append(normalized_data[i + sequence_length])
        
        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Force CPU execution to avoid CuDNN issues
        import tensorflow as tf
        import os
        
        # Disable GPU for LSTM to avoid CuDNN issues
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        
        # Build and train LSTM model on CPU only
        model = Sequential([
            LSTM(32, activation='tanh', recurrent_activation='sigmoid', 
                 input_shape=(sequence_length, 1), recurrent_dropout=0.0),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        
        # Train quickly with minimal epochs
        model.fit(X, y, epochs=3, verbose=0, batch_size=16)
        
        # Restore GPU visibility for other operations
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            del os.environ['CUDA_VISIBLE_DEVICES']
        
        # Fill gaps
        gap_indices = np.where(gap_mask)[0]
        for gap_idx in gap_indices:
            # Use preceding sequence to predict
            start_seq = max(0, gap_idx - sequence_length)
            if start_seq >= 0 and not np.isnan(filled_data.iloc[start_seq:gap_idx]).any():
                sequence = filled_data.iloc[start_seq:gap_idx].values[-sequence_length:]
                if len(sequence) == sequence_length:
                    seq_normalized = scaler.transform(sequence.reshape(-1, 1)).flatten()
                    prediction = model.predict(seq_normalized.reshape(1, -1, 1), verbose=0)
                    prediction_denorm = scaler.inverse_transform(prediction.reshape(-1, 1))[0, 0]
                    filled_data.iloc[gap_idx] = prediction_denorm
        
        return filled_data
    
    @staticmethod
    def transformer_fill(data, column, gap_mask, sequence_length=30):
        """Optimized transformer-based gap filling with performance improvements"""
        filled_data = data[column].copy()
        
        # Prepare training data from non-gap regions
        non_gap_data = filled_data.dropna().values
        
        if len(non_gap_data) < sequence_length * 2:
            return filled_data
        
        # Normalize data
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(non_gap_data.reshape(-1, 1)).flatten()
        
        # Create sequences with masking (reduced for speed)
        sequences = []
        targets = []
        # Use every 3rd sequence to speed up training
        for i in range(0, len(normalized_data) - sequence_length, 3):
            seq = normalized_data[i:i + sequence_length]
            target = normalized_data[i + sequence_length]
            sequences.append(seq)
            targets.append(target)
        
        if len(sequences) < 10:
            return filled_data
        
        # Move to GPU if available for speed
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        sequences = torch.tensor(sequences, dtype=torch.float32).to(device)
        targets = torch.tensor(targets, dtype=torch.float32).to(device)
        
        # Optimized lightweight transformer model
        class FastTransformer(nn.Module):
            def __init__(self, seq_len, d_model=32, nhead=2, num_layers=1):
                super().__init__()
                self.d_model = d_model
                self.input_projection = nn.Linear(1, d_model)
                self.positional_encoding = nn.Parameter(torch.randn(seq_len, d_model) * 0.1)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model, nhead, dim_feedforward=64, batch_first=True, dropout=0.1
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
                self.output_projection = nn.Linear(d_model, 1)
                
            def forward(self, x):
                x = x.unsqueeze(-1)  # Add feature dimension
                x = self.input_projection(x)
                x = x + self.positional_encoding[:x.size(1)]
                x = self.transformer(x)
                x = self.output_projection(x[:, -1, :])  # Use last token
                return x.squeeze(-1)
        
        model = FastTransformer(sequence_length).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
        criterion = nn.MSELoss()
        
        # Fast training with fewer epochs
        model.train()
        batch_size = min(64, len(sequences))
        for epoch in range(8):  # Reduced from 20
            # Mini-batch training
            for i in range(0, len(sequences), batch_size):
                batch_seq = sequences[i:i+batch_size]
                batch_targets = targets[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = model(batch_seq)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                optimizer.step()
        
        # Fill gaps
        model.eval()
        gap_indices = np.where(gap_mask)[0]
        for gap_idx in gap_indices:
            start_seq = max(0, gap_idx - sequence_length)
            if start_seq >= 0 and not np.isnan(filled_data.iloc[start_seq:gap_idx]).any():
                sequence = filled_data.iloc[start_seq:gap_idx].values[-sequence_length:]
                if len(sequence) == sequence_length:
                    seq_normalized = scaler.transform(sequence.reshape(-1, 1)).flatten()
                    seq_tensor = torch.tensor(seq_normalized.reshape(1, -1), dtype=torch.float32).to(device)
                    with torch.no_grad():
                        prediction = model(seq_tensor)
                        prediction_denorm = scaler.inverse_transform(prediction.cpu().numpy().reshape(-1, 1))[0, 0]
                        filled_data.iloc[gap_idx] = prediction_denorm
        
        return filled_data
    
    @staticmethod
    def bayesian_fill(data, column, gap_mask):
        """Improved Bayesian gap filling with local context and proper smoothing"""
        filled_data = data[column].copy()
        
        # Find gap regions
        gap_indices = np.where(gap_mask)[0]
        if len(gap_indices) == 0:
            return filled_data
        
        # Group consecutive gaps
        gap_groups = []
        current_group = [gap_indices[0]]
        
        for i in range(1, len(gap_indices)):
            if gap_indices[i] - gap_indices[i-1] == 1:
                current_group.append(gap_indices[i])
            else:
                gap_groups.append(current_group)
                current_group = [gap_indices[i]]
        gap_groups.append(current_group)
        
        # Fill each gap group using local Bayesian inference
        for gap_group in gap_groups:
            start_gap, end_gap = gap_group[0], gap_group[-1]
            gap_length = len(gap_group)
            
            # Get local context around the gap
            context_size = max(30, gap_length * 2)  # Adaptive context size
            
            # Extract left and right contexts
            left_start = max(0, start_gap - context_size)
            left_context = filled_data[left_start:start_gap].dropna()
            
            right_end = min(len(data), end_gap + context_size + 1)
            right_context = filled_data[end_gap + 1:right_end].dropna()
            
            # Combine local contexts for training
            local_data = pd.concat([left_context, right_context]).values
            
            if len(local_data) < 10:
                # Insufficient local data, fall back to global statistics
                global_data = filled_data.dropna().values
                if len(global_data) < 10:
                    # Use series mean as fallback
                    series_mean = np.nanmean(data[column].values)
                    filled_data.iloc[gap_group] = series_mean
                    continue
                local_data = global_data
            
            # Normalize local data
            scaler = MinMaxScaler()
            normalized_data = scaler.fit_transform(local_data.reshape(-1, 1)).flatten()
            
            # Simple Bayesian model with local adaptation
            def model(data_obs):
                # Adaptive priors based on local data statistics
                local_mean = float(np.mean(normalized_data))
                local_std = float(np.std(normalized_data)) + 1e-6  # Avoid zero std
                
                # Prior for mean (centered on local mean)
                mu = pyro.sample("mu", dist.Normal(local_mean, local_std))
                
                # Prior for std (based on local variability)
                sigma = pyro.sample("sigma", dist.Exponential(1.0 / local_std))
                
                # Likelihood
                with pyro.plate("data", len(data_obs)):
                    pyro.sample("obs", dist.Normal(mu, sigma), obs=data_obs)
            
            def guide(data_obs):
                # Variational parameters
                mu_q = pyro.param("mu_q", torch.tensor(float(np.mean(normalized_data))))
                sigma_q = pyro.param("sigma_q", torch.tensor(float(np.std(normalized_data)) + 0.1), 
                                   constraint=dist.constraints.positive)
                
                pyro.sample("mu", dist.Normal(mu_q, 0.1))
                pyro.sample("sigma", dist.LogNormal(torch.log(sigma_q), 0.1))
            
            # Clear previous parameters
            pyro.clear_param_store()
            
            # Set up inference
            svi = SVI(model, guide, Adam({"lr": 0.02}), loss=Trace_ELBO())
            
            # Train with local data
            data_tensor = torch.tensor(normalized_data, dtype=torch.float32)
            for step in range(50):  # Reduced iterations for speed
                loss = svi.step(data_tensor)
            
            # Get posterior parameters
            mu_post = pyro.param("mu_q").item()
            sigma_post = pyro.param("sigma_q").item()
            
            # Get boundary values for smoothing
            left_boundary = left_context.iloc[-1] if len(left_context) > 0 else mu_post
            right_boundary = right_context.iloc[0] if len(right_context) > 0 else mu_post
            
            # Normalize boundaries for blending
            left_boundary_norm = scaler.transform([[left_boundary]])[0, 0] if len(left_context) > 0 else mu_post
            right_boundary_norm = scaler.transform([[right_boundary]])[0, 0] if len(right_context) > 0 else mu_post
            
            # Generate smooth predictions for gap
            gap_predictions = []
            
            for i, gap_idx in enumerate(gap_group):
                # Position weight (0 at start of gap, 1 at end of gap)
                position_weight = i / (gap_length - 1) if gap_length > 1 else 0.5
                
                # Sample multiple predictions and take stable average
                samples = np.random.normal(mu_post, sigma_post * 0.5, 20)  # Reduced variance for stability
                
                # Weight samples by their likelihood under the posterior
                log_probs = -0.5 * ((samples - mu_post) / sigma_post) ** 2
                weights = np.exp(log_probs - np.max(log_probs))
                weights /= np.sum(weights)
                
                # Weighted prediction
                bayesian_pred = np.sum(weights * samples)
                
                # Interpolated boundary prediction for smoothness
                boundary_pred = (1 - position_weight) * left_boundary_norm + position_weight * right_boundary_norm
                
                # Blend Bayesian and boundary predictions
                # More boundary influence for short gaps, more Bayesian for longer gaps
                boundary_weight = max(0.2, 1.0 - gap_length / 20.0)
                bayesian_weight = 1.0 - boundary_weight
                
                final_pred_norm = boundary_weight * boundary_pred + bayesian_weight * bayesian_pred
                
                # Denormalize
                final_pred = scaler.inverse_transform([[final_pred_norm]])[0, 0]
                gap_predictions.append(final_pred)
            
            # Apply additional smoothing to reduce noise
            if gap_length > 2:
                # Simple moving average to smooth predictions
                gap_predictions = np.array(gap_predictions)
                smoothed = np.copy(gap_predictions)
                
                for i in range(1, gap_length - 1):
                    smoothed[i] = 0.25 * gap_predictions[i-1] + 0.5 * gap_predictions[i] + 0.25 * gap_predictions[i+1]
                
                gap_predictions = smoothed
            
            # Assign predictions to gap
            filled_data.iloc[gap_group] = gap_predictions
        
        return filled_data

def display_method_documentation(method):
    """Display mathematical documentation for each gap filling method"""
    if method == "None (Raw Data)":
        st.sidebar.markdown("""
        ### Raw Data Display
        Shows the original time series with missing values as gaps.
        
        **Mathematical Definition:**
        
        $x(t) = \\begin{cases} 
        x_{observed}(t) & \\text{if } t \\in T_{observed} \\\\
        \\text{NaN} & \\text{if } t \\in T_{missing}
        \\end{cases}$
        
        Where $T_{observed}$ and $T_{missing}$ are the sets of observed and missing time indices.
        """)
    
    elif method == "Linear Interpolation":
        st.sidebar.markdown("""
        ### Linear Interpolation
        
        **Mathematical Definition:**
        
        For a gap between points $(t_i, x_i)$ and $(t_j, x_j)$:
        
        $x(t) = x_i + \\frac{x_j - x_i}{t_j - t_i}(t - t_i)$
        
        **Properties:**
        - ✅ Perfect continuity at boundaries
        - ✅ Computationally efficient: $O(n)$  
        - ✅ Stable for all gap lengths
        - ⚠️ Assumes linear trend (no frequency preservation)
        
        **Best For:** All gap types, baseline method
        """)
    
    elif method == "FFT Reconstruction":
        st.sidebar.markdown("""
        ### FFT-Based Reconstruction
        
        **Mathematical Definition:**
        
        1. **Context Extraction:** $x_{ctx} = [x_{t-k}, ..., x_{t+k}] \\setminus \\text{gaps}$
        
        2. **Frequency Analysis:** $X(f) = \\mathcal{F}\\{x_{ctx}\\} = \\sum_{n} x_{ctx}[n] e^{-2\\pi i fn}$
        
        3. **Frequency Selection:** $F_{top} = \\arg\\max_{f} |X(f)|$ (top 3 components)
        
        4. **Reconstruction:** $x_{gap}(t) = \\sum_{f \\in F_{top}} \\frac{w_f \\cdot X(f)}{|x_{ctx}|} e^{2\\pi i ft}$
        
        5. **Boundary Blending:** $x_{final} = \\alpha \\cdot \\text{linear}(t) + (1-\\alpha) \\cdot \\text{Re}\\{x_{gap}(t)\\}$
        
        **Properties:**
        - ✅ Preserves frequency characteristics
        - ✅ Good for periodic signals
        - ✅ Spike-free (scaled amplitudes)
        - ⚠️ Requires sufficient context data
        
        **Best For:** Medium gaps (5-20 samples) with periodic patterns
        """)
    
    elif method == "LSTM Prediction":
        st.sidebar.markdown("""
        ### LSTM Neural Network
        
        **Mathematical Definition:**
        
        **LSTM Cell Equations:**
        
        $f_t = \\sigma(W_f \\cdot [h_{t-1}, x_t] + b_f)$ (forget gate)
        
        $i_t = \\sigma(W_i \\cdot [h_{t-1}, x_t] + b_i)$ (input gate)
        
        $\\tilde{C}_t = \\tanh(W_C \\cdot [h_{t-1}, x_t] + b_C)$ (candidate)
        
        $C_t = f_t * C_{t-1} + i_t * \\tilde{C}_t$ (cell state)
        
        $o_t = \\sigma(W_o \\cdot [h_{t-1}, x_t] + b_o)$ (output gate)
        
        $h_t = o_t * \\tanh(C_t)$ (hidden state)
        
        **Prediction:** $\\hat{x}_{t+1} = W_{out} \\cdot h_t + b_{out}$
        
        **Properties:**
        - ✅ Captures long-term dependencies
        - ✅ Non-linear pattern modeling
        - ⚠️ Requires training data
        - ⚠️ CPU-only (CuDNN compatibility)
        
        **Best For:** Complex patterns, sequential dependencies
        """)
    
    elif method == "Transformer":
        st.sidebar.markdown("""
        ### Transformer Architecture
        
        **Mathematical Definition:**
        
        **Self-Attention Mechanism:**
        
        $\\text{Attention}(Q,K,V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V$
        
        **Multi-Head Attention:**
        
        $\\text{MultiHead}(Q,K,V) = \\text{Concat}(\\text{head}_1, ..., \\text{head}_h)W^O$
        
        $\\text{head}_i = \\text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$
        
        **Positional Encoding:**
        
        $PE_{(pos,2i)} = \\sin\\left(\\frac{pos}{10000^{2i/d}}\\right)$
        
        $PE_{(pos,2i+1)} = \\cos\\left(\\frac{pos}{10000^{2i/d}}\\right)$
        
        **Properties:**
        - ✅ Parallel processing
        - ✅ Attention-based context modeling
        - ✅ Fast inference (optimized)
        - ⚠️ Requires sufficient training data
        
        **Best For:** Complex patterns, attention-based modeling
        """)
    
    elif method == "Bayesian + Smoothing":
        st.sidebar.markdown("""
        ### Bayesian Inference with Local Smoothing
        
        **Mathematical Definition:**
        
        **Local Context Model:**
        
        $x_{local} \\sim \\mathcal{N}(\\mu_{local}, \\sigma_{local})$
        
        **Bayesian Prior:** $\\mu \\sim \\mathcal{N}(\\hat{\\mu}_{local}, \\hat{\\sigma}_{local})$, $\\sigma \\sim \\text{Exp}(1/\\hat{\\sigma}_{local})$
        
        **Posterior Update:**
        
        $p(\\mu, \\sigma | x_{local}) \\propto p(x_{local} | \\mu, \\sigma) \\cdot p(\\mu) \\cdot p(\\sigma)$
        
        **Gap Prediction:**
        
        $x_{gap}(i) = w_{boundary} \\cdot x_{linear}(i) + w_{bayes} \\cdot \\mathbb{E}[x | \\mu_{post}, \\sigma_{post}]$
        
        **Smoothing Filter:**
        
        $x_{smooth}[i] = 0.25 \\cdot x[i-1] + 0.5 \\cdot x[i] + 0.25 \\cdot x[i+1]$
        
        **Properties:**
        - ✅ Uncertainty quantification
        - ✅ Local context adaptation
        - ✅ Smooth boundary transitions
        - ✅ Excellent for short gaps
        
        **Best For:** Short to medium gaps (3-15 samples)
        """)

def main():
    st.set_page_config(page_title="Solar Wind Gap Filling", layout="wide")
    st.title("Solar Wind Time Series Gap Filling Methods")
    
    # Initialize data
    if 'solar_data' not in st.session_state:
        st.session_state.solar_data = SolarWindData()
        st.session_state.data_with_gaps = st.session_state.solar_data.create_gaps()
    
    # Sidebar controls
    st.sidebar.header("Controls")
    
    # Select time range for display
    time_range = st.sidebar.slider(
        "Time Range (hours from start)",
        min_value=0,
        max_value=24*7,  # 1 week
        value=(0, 24),  # Default to first 24 hours
        step=1
    )
    
    # Select method
    method = st.sidebar.selectbox(
        "Gap Filling Method",
        ["None (Raw Data)", "Linear Interpolation", "FFT Reconstruction", "LSTM Prediction", "Transformer", "Bayesian + Smoothing"]
    )
    
    # Display mathematical documentation for selected method
    st.sidebar.markdown("---")
    display_method_documentation(method)
    
    # Get data slice
    start_idx = time_range[0] * 60
    end_idx = time_range[1] * 60
    data_slice = st.session_state.data_with_gaps.iloc[start_idx:end_idx].copy()
    original_slice = st.session_state.solar_data.data.iloc[start_idx:end_idx].copy()
    
    # Apply gap filling method
    if method != "None (Raw Data)":
        filled_data = data_slice.copy()
        
        for column in data_slice.columns:
            gap_mask = data_slice[column].isna()
            
            if method == "Linear Interpolation":
                filled_data[column] = GapFillingMethods.interpolation_fill(data_slice, 'linear')[column]
            elif method == "FFT Reconstruction":
                filled_data[column] = GapFillingMethods.fft_fill(data_slice, column, gap_mask)
            elif method == "LSTM Prediction":
                filled_data[column] = GapFillingMethods.lstm_fill(data_slice, column, gap_mask)
            elif method == "Transformer":
                filled_data[column] = GapFillingMethods.transformer_fill(data_slice, column, gap_mask)
            elif method == "Bayesian + Smoothing":
                filled_data[column] = GapFillingMethods.bayesian_fill(data_slice, column, gap_mask)
    else:
        filled_data = data_slice.copy()
    
    # Create plots
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("B Vector (Magnetic Field)")
        fig_b = make_subplots(
            rows=3, cols=1,
            subplot_titles=['Bx', 'By', 'Bz'],
            shared_xaxes=True
        )
        
        for i, component in enumerate(['Bx', 'By', 'Bz']):
            # Original data
            fig_b.add_trace(
                go.Scatter(
                    x=original_slice.index,
                    y=original_slice[component],
                    name=f'{component} Original',
                    line=dict(color='blue', width=1),
                    opacity=0.7
                ),
                row=i+1, col=1
            )
            
            # Data with gaps
            fig_b.add_trace(
                go.Scatter(
                    x=data_slice.index,
                    y=data_slice[component],
                    name=f'{component} With Gaps',
                    mode='markers',
                    marker=dict(color='red', size=2)
                ),
                row=i+1, col=1
            )
            
            # Filled data (if method applied)
            if method != "None (Raw Data)":
                fig_b.add_trace(
                    go.Scatter(
                        x=filled_data.index,
                        y=filled_data[component],
                        name=f'{component} Filled',
                        line=dict(color='green', width=2)
                    ),
                    row=i+1, col=1
                )
        
        fig_b.update_layout(height=600, showlegend=True)
        st.plotly_chart(fig_b, use_container_width=True)
    
    with col2:
        st.subheader("V Vector (Velocity)")
        fig_v = make_subplots(
            rows=3, cols=1,
            subplot_titles=['Vx', 'Vy', 'Vz'],
            shared_xaxes=True
        )
        
        for i, component in enumerate(['Vx', 'Vy', 'Vz']):
            # Original data
            fig_v.add_trace(
                go.Scatter(
                    x=original_slice.index,
                    y=original_slice[component],
                    name=f'{component} Original',
                    line=dict(color='blue', width=1),
                    opacity=0.7
                ),
                row=i+1, col=1
            )
            
            # Data with gaps
            fig_v.add_trace(
                go.Scatter(
                    x=data_slice.index,
                    y=data_slice[component],
                    name=f'{component} With Gaps',
                    mode='markers',
                    marker=dict(color='red', size=2)
                ),
                row=i+1, col=1
            )
            
            # Filled data (if method applied)
            if method != "None (Raw Data)":
                fig_v.add_trace(
                    go.Scatter(
                        x=filled_data.index,
                        y=filled_data[component],
                        name=f'{component} Filled',
                        line=dict(color='green', width=2)
                    ),
                    row=i+1, col=1
                )
        
        fig_v.update_layout(height=600, showlegend=True)
        st.plotly_chart(fig_v, use_container_width=True)
    
    # 3D Vector plots
    st.subheader("3D Vector Visualization")
    col3, col4 = st.columns(2)
    
    with col3:
        st.write("**B Vector 3D**")
        fig_3d_b = go.Figure(data=go.Scatter3d(
            x=filled_data['Bx'] if method != "None (Raw Data)" else data_slice['Bx'],
            y=filled_data['By'] if method != "None (Raw Data)" else data_slice['By'],
            z=filled_data['Bz'] if method != "None (Raw Data)" else data_slice['Bz'],
            mode='markers',
            marker=dict(
                size=2,
                color=np.arange(len(data_slice)),
                colorscale='Viridis',
                showscale=True
            )
        ))
        fig_3d_b.update_layout(scene=dict(
            xaxis_title='Bx',
            yaxis_title='By',
            zaxis_title='Bz'
        ))
        st.plotly_chart(fig_3d_b, use_container_width=True)
    
    with col4:
        st.write("**V Vector 3D**")
        fig_3d_v = go.Figure(data=go.Scatter3d(
            x=filled_data['Vx'] if method != "None (Raw Data)" else data_slice['Vx'],
            y=filled_data['Vy'] if method != "None (Raw Data)" else data_slice['Vy'],
            z=filled_data['Vz'] if method != "None (Raw Data)" else data_slice['Vz'],
            mode='markers',
            marker=dict(
                size=2,
                color=np.arange(len(data_slice)),
                colorscale='Plasma',
                showscale=True
            )
        ))
        fig_3d_v.update_layout(scene=dict(
            xaxis_title='Vx',
            yaxis_title='Vy',
            zaxis_title='Vz'
        ))
        st.plotly_chart(fig_3d_v, use_container_width=True)
    
    # Statistics
    st.subheader("Gap Statistics")
    gap_stats = {}
    for column in data_slice.columns:
        gaps = data_slice[column].isna().sum()
        gap_stats[column] = f"{gaps} samples ({gaps/len(data_slice)*100:.1f}%)"
    
    st.write(gap_stats)

if __name__ == "__main__":
    main()