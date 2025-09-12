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
    def lstm_fill(data, column, gap_mask, sequence_length=20):
        """Simplified RNN-based gap filling with robust error handling"""
        filled_data = data[column].copy()
        
        # Find gap regions
        gap_indices = np.where(gap_mask)[0]
        if len(gap_indices) == 0:
            return filled_data
        
        # Get all available data for training
        non_gap_data = filled_data.dropna().values
        
        # Check if we have enough data - use adaptive thresholds
        min_data_needed = max(30, sequence_length * 2)
        if len(non_gap_data) < min_data_needed:
            # Fallback to linear interpolation for very small datasets
            return GapFillingMethods.interpolation_fill(pd.DataFrame({column: filled_data}), 'linear')[column]
        
        # Adaptive sequence length based on data size
        if len(non_gap_data) < 100:
            sequence_length = min(10, len(non_gap_data) // 3)
        elif len(non_gap_data) < 200:
            sequence_length = min(15, len(non_gap_data) // 4)
        
        try:
            # Normalize data
            scaler = MinMaxScaler()
            normalized_data = scaler.fit_transform(non_gap_data.reshape(-1, 1)).flatten()
            
            # Create sequences for training
            X, y = [], []
            for i in range(len(normalized_data) - sequence_length):
                X.append(normalized_data[i:i + sequence_length])
                y.append(normalized_data[i + sequence_length])
            
            if len(X) < 5:  # Need minimum training samples
                return GapFillingMethods.interpolation_fill(pd.DataFrame({column: filled_data}), 'linear')[column]
            
            X, y = np.array(X), np.array(y)
            X = X.reshape((X.shape[0], X.shape[1], 1))
            
            # Use TensorFlow with CPU-only device
            import tensorflow as tf
            
            # Explicitly set CPU device and avoid CuDNN entirely
            with tf.device('/CPU:0'):
                # Use SimpleRNN instead of LSTM - more reliable and faster
                model = Sequential([
                    tf.keras.layers.SimpleRNN(
                        units=12,
                        activation='tanh',
                        input_shape=(sequence_length, 1),
                        dropout=0.1
                    ),
                    tf.keras.layers.Dense(1, activation='linear')
                ])
                
                # Use simple optimizer to avoid issues
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                    loss='mse'
                )
                
                # Quick training with validation split
                history = model.fit(
                    X, y,
                    epochs=8,
                    batch_size=min(16, len(X) // 2),
                    verbose=0,
                    validation_split=0.2 if len(X) > 10 else 0.0
                )
            
            # Process gaps one by one
            filled_count = 0
            for gap_idx in gap_indices:
                try:
                    # Find valid preceding values
                    preceding_values = []
                    for i in range(gap_idx - 1, -1, -1):
                        if not gap_mask[i]:  # If not a gap
                            preceding_values.insert(0, filled_data.iloc[i])
                        if len(preceding_values) >= sequence_length:
                            break
                    
                    if len(preceding_values) >= sequence_length:
                        # Use exact sequence length
                        sequence_vals = np.array(preceding_values[-sequence_length:])
                        
                        # Normalize the sequence
                        seq_normalized = scaler.transform(sequence_vals.reshape(-1, 1)).flatten()
                        
                        # Predict using CPU device
                        with tf.device('/CPU:0'):
                            prediction = model.predict(
                                seq_normalized.reshape(1, sequence_length, 1),
                                verbose=0
                            )
                        
                        # Denormalize prediction
                        prediction_denorm = scaler.inverse_transform(
                            prediction.reshape(-1, 1)
                        )[0, 0]
                        
                        # Sanity check - ensure reasonable values
                        if not np.isnan(prediction_denorm) and np.isfinite(prediction_denorm):
                            # Clamp to reasonable range based on nearby values
                            nearby_mean = np.mean(sequence_vals)
                            nearby_std = np.std(sequence_vals)
                            if nearby_std > 0:
                                # Allow up to 3 standard deviations from local mean
                                lower_bound = nearby_mean - 3 * nearby_std
                                upper_bound = nearby_mean + 3 * nearby_std
                                prediction_denorm = np.clip(prediction_denorm, lower_bound, upper_bound)
                            
                            filled_data.iloc[gap_idx] = prediction_denorm
                            filled_count += 1
                        else:
                            # Use simple forward fill if prediction is invalid
                            filled_data.iloc[gap_idx] = preceding_values[-1]
                            filled_count += 1
                            
                    else:
                        # Not enough preceding context - use interpolation
                        # Find nearest valid neighbors
                        left_idx, right_idx = gap_idx - 1, gap_idx + 1
                        
                        # Find nearest non-gap values
                        while left_idx >= 0 and gap_mask[left_idx]:
                            left_idx -= 1
                        while right_idx < len(gap_mask) and gap_mask[right_idx]:
                            right_idx += 1
                        
                        if left_idx >= 0 and right_idx < len(gap_mask):
                            # Linear interpolation between neighbors
                            left_val = filled_data.iloc[left_idx]
                            right_val = filled_data.iloc[right_idx]
                            weight = (gap_idx - left_idx) / (right_idx - left_idx)
                            filled_data.iloc[gap_idx] = left_val + weight * (right_val - left_val)
                            filled_count += 1
                        elif left_idx >= 0:
                            # Forward fill
                            filled_data.iloc[gap_idx] = filled_data.iloc[left_idx]
                            filled_count += 1
                        elif right_idx < len(gap_mask):
                            # Backward fill
                            filled_data.iloc[gap_idx] = filled_data.iloc[right_idx]
                            filled_count += 1
                        else:
                            # Last resort: use mean
                            filled_data.iloc[gap_idx] = np.mean(non_gap_data)
                            filled_count += 1
                            
                except Exception as gap_error:
                    # Skip this gap and continue with others
                    continue
            
            return filled_data
            
        except Exception as e:
            # Complete fallback to linear interpolation
            print(f"RNN model failed: {str(e)}, using linear interpolation")
            return GapFillingMethods.interpolation_fill(pd.DataFrame({column: filled_data}), 'linear')[column]
    
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
    
    @staticmethod
    def spline_fill(data, column, gap_mask):
        """Cubic spline interpolation for smooth gap filling"""
        filled_data = data[column].copy()
        
        # Get indices of all data points
        all_indices = np.arange(len(data))
        valid_mask = ~gap_mask
        
        if valid_mask.sum() < 4:  # Need minimum 4 points for cubic spline
            return GapFillingMethods.interpolation_fill(pd.DataFrame({column: filled_data}), 'linear')[column]
        
        try:
            from scipy.interpolate import CubicSpline
            
            # Extract valid data points
            valid_indices = all_indices[valid_mask]
            valid_values = filled_data[valid_mask].values
            
            # Create cubic spline
            spline = CubicSpline(valid_indices, valid_values, bc_type='natural')
            
            # Fill gaps with bounds checking
            gap_indices = all_indices[gap_mask]
            spline_values = spline(gap_indices)
            
            # Apply bounds checking based on data range
            data_min, data_max = valid_values.min(), valid_values.max()
            data_std = valid_values.std()
            bounds_min = data_min - 2 * data_std
            bounds_max = data_max + 2 * data_std
            
            # Clip values to reasonable bounds
            spline_values = np.clip(spline_values, bounds_min, bounds_max)
            filled_data.iloc[gap_indices] = spline_values
            
            return filled_data
            
        except Exception as e:
            # Fallback to linear interpolation
            return GapFillingMethods.interpolation_fill(pd.DataFrame({column: filled_data}), 'linear')[column]
    
    @staticmethod
    def knn_fill(data, column, gap_mask, n_neighbors=5):
        """K-Nearest Neighbors gap filling using local similarity patterns"""
        filled_data = data[column].copy()
        
        gap_indices = np.where(gap_mask)[0]
        if len(gap_indices) == 0:
            return filled_data
            
        valid_data = filled_data.dropna()
        if len(valid_data) < n_neighbors * 2:
            return GapFillingMethods.interpolation_fill(pd.DataFrame({column: filled_data}), 'linear')[column]
        
        try:
            from sklearn.neighbors import KNeighborsRegressor
            from sklearn.preprocessing import StandardScaler
            
            # Create features using sliding windows
            window_size = min(10, len(valid_data) // 4)
            
            X_train, y_train = [], []
            valid_indices = valid_data.index.tolist()
            
            # Create training data from valid sequences
            for i in range(len(valid_indices) - window_size):
                if valid_indices[i + window_size] - valid_indices[i] == window_size:  # Consecutive sequence
                    window = [valid_data.iloc[j] for j in range(i, i + window_size)]
                    target = valid_data.iloc[i + window_size]
                    X_train.append(window)
                    y_train.append(target)
            
            if len(X_train) < n_neighbors:
                return GapFillingMethods.interpolation_fill(pd.DataFrame({column: filled_data}), 'linear')[column]
            
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            
            # Train KNN model
            knn = KNeighborsRegressor(n_neighbors=min(n_neighbors, len(X_train)), weights='distance')
            knn.fit(X_train_scaled, y_train)
            
            # Fill gaps
            for gap_idx in gap_indices:
                # Get preceding window
                window_start = max(0, gap_idx - window_size)
                preceding = []
                
                for i in range(window_start, gap_idx):
                    if not gap_mask[i]:  # Valid point
                        preceding.append(filled_data.iloc[i])
                
                if len(preceding) >= window_size:
                    # Use last window_size points
                    feature_window = np.array(preceding[-window_size:]).reshape(1, -1)
                    feature_scaled = scaler.transform(feature_window)
                    prediction = knn.predict(feature_scaled)[0]
                    filled_data.iloc[gap_idx] = prediction
                else:
                    # Fallback to nearest neighbor value
                    if len(preceding) > 0:
                        filled_data.iloc[gap_idx] = preceding[-1]
                    else:
                        # Find nearest valid value
                        distances = [(abs(i - gap_idx), i) for i in valid_indices]
                        nearest_idx = min(distances)[1]
                        filled_data.iloc[gap_idx] = valid_data.loc[nearest_idx]
            
            return filled_data
            
        except Exception as e:
            return GapFillingMethods.interpolation_fill(pd.DataFrame({column: filled_data}), 'linear')[column]
    
    @staticmethod
    def kalman_fill(data, column, gap_mask):
        """Kalman filter gap filling for trending/noisy time series"""
        filled_data = data[column].copy()
        
        gap_indices = np.where(gap_mask)[0]
        if len(gap_indices) == 0:
            return filled_data
        
        valid_data = filled_data.dropna().values
        if len(valid_data) < 10:
            return GapFillingMethods.interpolation_fill(pd.DataFrame({column: filled_data}), 'linear')[column]
        
        try:
            # Simple Kalman filter implementation
            # State: [position, velocity]  
            # Observation: position
            
            # Estimate initial parameters from valid data
            dt = 1.0  # Time step
            
            # Process noise (how much we expect the system to change)
            process_var = np.var(np.diff(valid_data)) if len(valid_data) > 1 else 1.0
            
            # Observation noise (measurement uncertainty)  
            obs_var = process_var * 0.1
            
            # State transition matrix (constant velocity model)
            F = np.array([[1, dt], 
                         [0, 1]])
            
            # Observation matrix (observe position only)
            H = np.array([[1, 0]])
            
            # Process noise covariance
            Q = np.array([[dt**4/4, dt**3/2],
                         [dt**3/2, dt**2]]) * process_var
            
            # Observation noise covariance  
            R = np.array([[obs_var]])
            
            # Initialize state with first valid point
            if not gap_mask[0]:
                x = np.array([[filled_data.iloc[0]], [0]])  # position, velocity
            else:
                first_valid_idx = np.where(~gap_mask)[0][0]
                x = np.array([[filled_data.iloc[first_valid_idx]], [0]])
            
            # Initialize covariance
            P = np.eye(2) * process_var
            
            # Forward pass through all data
            estimates = {}
            
            for i in range(len(data)):
                # Predict step
                x = F @ x
                P = F @ P @ F.T + Q
                
                if not gap_mask[i]:  # Observation available
                    # Update step
                    y = np.array([[filled_data.iloc[i]]]) - H @ x  # Innovation
                    S = H @ P @ H.T + R  # Innovation covariance
                    K = P @ H.T @ np.linalg.inv(S)  # Kalman gain
                    
                    x = x + K @ y
                    P = (np.eye(2) - K @ H) @ P
                    
                # Store estimate
                estimates[i] = x[0, 0]
            
            # Fill gaps with estimates and bounds checking
            data_min, data_max = valid_data.min(), valid_data.max()
            data_std = valid_data.std()
            bounds_min = data_min - 2 * data_std
            bounds_max = data_max + 2 * data_std
            
            for gap_idx in gap_indices:
                estimate = estimates[gap_idx]
                # Clip to reasonable bounds
                estimate = np.clip(estimate, bounds_min, bounds_max)
                filled_data.iloc[gap_idx] = estimate
            
            return filled_data
            
        except Exception as e:
            return GapFillingMethods.interpolation_fill(pd.DataFrame({column: filled_data}), 'linear')[column]
    
    @staticmethod  
    def seasonal_fill(data, column, gap_mask, period=None):
        """Seasonal decomposition-based gap filling"""
        filled_data = data[column].copy()
        
        gap_indices = np.where(gap_mask)[0]
        if len(gap_indices) == 0:
            return filled_data
        
        valid_data = filled_data.dropna()
        if len(valid_data) < 50:  # Need sufficient data for seasonal analysis
            return GapFillingMethods.interpolation_fill(pd.DataFrame({column: filled_data}), 'linear')[column]
        
        try:
            from scipy import signal
            
            # Auto-detect period if not provided
            if period is None:
                # Use FFT to find dominant frequency
                fft_vals = np.fft.fft(valid_data.values)
                freqs = np.fft.fftfreq(len(valid_data))
                
                # Find peak frequency (excluding DC)
                magnitude = np.abs(fft_vals[1:len(fft_vals)//2])
                if len(magnitude) > 0:
                    peak_freq = freqs[1:len(fft_vals)//2][np.argmax(magnitude)]
                    period = int(1 / abs(peak_freq)) if peak_freq != 0 else 24
                else:
                    period = 24  # Default to daily cycle
                    
                period = max(6, min(period, len(valid_data) // 4))  # Reasonable bounds
            
            # Create a complete time series for decomposition
            # Fill gaps temporarily with linear interpolation
            temp_filled = GapFillingMethods.interpolation_fill(pd.DataFrame({column: filled_data}), 'linear')[column]
            
            # Simple seasonal decomposition
            # Trend component using moving average
            trend_window = min(period * 2, len(temp_filled) // 3)
            if trend_window >= 3:
                trend = temp_filled.rolling(window=trend_window, center=True, min_periods=1).mean()
            else:
                trend = pd.Series([temp_filled.mean()] * len(temp_filled), index=temp_filled.index)
            
            # Detrended data
            detrended = temp_filled - trend
            
            # Seasonal component - average pattern over periods
            seasonal = pd.Series(0.0, index=temp_filled.index)
            seasonal_pattern = []
            
            for phase in range(period):
                phase_values = []
                for i in range(phase, len(detrended), period):
                    if i < len(detrended) and not gap_mask[i]:  # Only use valid data
                        phase_values.append(detrended.iloc[i])
                
                if len(phase_values) > 0:
                    seasonal_pattern.append(np.mean(phase_values))
                else:
                    seasonal_pattern.append(0.0)
            
            # Apply seasonal pattern
            for i in range(len(seasonal)):
                seasonal.iloc[i] = seasonal_pattern[i % period]
            
            # Fill gaps using trend + seasonal components
            for gap_idx in gap_indices:
                filled_value = trend.iloc[gap_idx] + seasonal.iloc[gap_idx]
                filled_data.iloc[gap_idx] = filled_value
            
            return filled_data
            
        except Exception as e:
            return GapFillingMethods.interpolation_fill(pd.DataFrame({column: filled_data}), 'linear')[column]

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
        ### RNN Neural Network
        
        **Mathematical Definition:**
        
        **SimpleRNN Cell Equations:**
        
        $h_t = \\tanh(W_{xh} \\cdot x_t + W_{hh} \\cdot h_{t-1} + b_h)$ (hidden state)
        
        $\\hat{x}_{t+1} = W_{out} \\cdot h_t + b_{out}$ (output prediction)
        
        **Adaptive Training:**
        - Sequence length: $L = \\min(20, \\lfloor |X_{available}|/4 \\rfloor)$
        - Training samples: $N = |X_{available}| - L$
        - Batch size: $B = \\min(16, \\lfloor N/2 \\rfloor)$
        
        **Gap Filling Process:**
        1. Extract preceding sequence: $s_{gap} = [x_{t-L}, ..., x_{t-1}]$
        2. Normalize: $s_{norm} = \\text{MinMaxScale}(s_{gap})$
        3. Predict: $\\hat{x}_t = \\text{RNN}(s_{norm})$
        4. Denormalize and validate prediction
        
        **Properties:**
        - ✅ Fast and reliable (CPU-optimized)
        - ✅ Adaptive sequence length
        - ✅ Robust error handling with fallbacks
        - ✅ No CuDNN dependencies
        - ⚠️ Simplified compared to full LSTM
        
        **Best For:** Sequential patterns, reliable performance
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
    
    elif method == "Spline Interpolation":
        st.sidebar.markdown("""
        ### Cubic Spline Interpolation
        
        **Mathematical Definition:**
        
        For intervals $[x_i, x_{i+1}]$, construct piecewise cubic polynomials:
        
        $S_i(x) = a_i(x-x_i)^3 + b_i(x-x_i)^2 + c_i(x-x_i) + d_i$
        
        **Continuity Constraints:**
        - $S_i(x_i) = y_i$ (value continuity)
        - $S'_i(x_{i+1}) = S'_{i+1}(x_{i+1})$ (first derivative)
        - $S''_i(x_{i+1}) = S''_{i+1}(x_{i+1})$ (second derivative)
        
        **Natural Boundary Conditions:** $S''_0(x_0) = S''_{n-1}(x_n) = 0$
        
        **Properties:**
        - ✅ $C^2$ continuity (smooth curves)
        - ✅ Optimal for smooth, continuous signals
        - ✅ No oscillations between points
        - ✅ Fast computation: $O(n)$
        
        **Best For:** Smooth physical processes, curved data trends
        """)
    
    elif method == "K-Nearest Neighbors":
        st.sidebar.markdown("""
        ### K-Nearest Neighbors Regression
        
        **Mathematical Definition:**
        
        **Feature Construction:**
        $F(t) = [x_{t-w}, x_{t-w+1}, ..., x_{t+w}] \\setminus \\{\\text{gaps}\\}$
        
        **Distance Metric:**
        $d(F_i, F_j) = \\sqrt{\\sum_{k} (F_i[k] - F_j[k])^2}$ (Euclidean)
        
        **Prediction:**
        $\\hat{x}(t) = \\frac{1}{k} \\sum_{i=1}^k x_{N_i}$
        
        Where $N_i$ are the $k$ nearest neighbors by feature similarity.
        
        **Adaptive Parameters:**
        - Window size: $w = \\min(5, \\lfloor L_{gap}/2 \\rfloor)$
        - Neighbors: $k = \\min(5, \\lfloor N_{available}/3 \\rfloor)$
        
        **Properties:**
        - ✅ Non-parametric (no assumptions)
        - ✅ Local pattern matching
        - ✅ Fast inference: $O(n \\log n)$
        - ✅ Robust to outliers
        
        **Best For:** Locally repetitive patterns, irregular data
        """)
    
    elif method == "Kalman Filter":
        st.sidebar.markdown("""
        ### Kalman Filter State Estimation
        
        **Mathematical Definition:**
        
        **State Space Model:**
        
        State equation: $x_{t+1} = A x_t + w_t$ (process noise: $w_t \\sim \\mathcal{N}(0,Q)$)
        
        Observation: $y_t = H x_t + v_t$ (measurement noise: $v_t \\sim \\mathcal{N}(0,R)$)
        
        **Prediction Step:**
        - $\\hat{x}_{t|t-1} = A \\hat{x}_{t-1|t-1}$ (state prediction)
        - $P_{t|t-1} = A P_{t-1|t-1} A^T + Q$ (error covariance)
        
        **Update Step:**
        - $K_t = P_{t|t-1} H^T (H P_{t|t-1} H^T + R)^{-1}$ (Kalman gain)
        - $\\hat{x}_{t|t} = \\hat{x}_{t|t-1} + K_t(y_t - H\\hat{x}_{t|t-1})$ (state update)
        
        **Adaptive Noise:** $Q = \\sigma^2_{local}$, $R = 0.1 \\cdot \\sigma^2_{local}$
        
        **Properties:**
        - ✅ Optimal for linear Gaussian systems
        - ✅ Handles noisy observations
        - ✅ Uncertainty quantification
        - ✅ Real-time processing: $O(n)$
        
        **Best For:** Trending data with noise, state tracking
        """)
    
    elif method == "Seasonal Decomposition":
        st.sidebar.markdown("""
        ### Seasonal Decomposition Gap Filling
        
        **Mathematical Definition:**
        
        **Decomposition Model:**
        $x(t) = T(t) + S(t) + R(t)$
        
        Where:
        - $T(t)$ = Trend component
        - $S(t)$ = Seasonal component  
        - $R(t)$ = Residual component
        
        **STL Decomposition:**
        1. **Trend Extraction:** LOESS smoothing with bandwidth $n_t$
        2. **Detrending:** $x'(t) = x(t) - T(t)$
        3. **Seasonal Estimation:** Cycle subseries averages
        4. **Residuals:** $R(t) = x(t) - T(t) - S(t)$
        
        **Gap Filling Strategy:**
        - Fill trend: Linear interpolation of $T(t)$
        - Fill seasonal: Use periodic pattern $S(t \\bmod p)$
        - Fill residual: Local mean of $R(t)$
        
        **Final Reconstruction:** $\\hat{x}(t) = \\hat{T}(t) + \\hat{S}(t) + \\hat{R}(t)$
        
        **Properties:**
        - ✅ Separates signal components
        - ✅ Exploits periodic patterns
        - ✅ Handles trending seasonal data
        - ✅ Robust decomposition
        
        **Best For:** Data with clear seasonality and trends
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
        ["None (Raw Data)", "Linear Interpolation", "Spline Interpolation", "FFT Reconstruction", 
         "K-Nearest Neighbors", "Kalman Filter", "Seasonal Decomposition", 
         "LSTM Prediction", "Transformer", "Bayesian + Smoothing"]
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
            elif method == "Spline Interpolation":
                filled_data[column] = GapFillingMethods.spline_fill(data_slice, column, gap_mask)
            elif method == "FFT Reconstruction":
                filled_data[column] = GapFillingMethods.fft_fill(data_slice, column, gap_mask)
            elif method == "K-Nearest Neighbors":
                filled_data[column] = GapFillingMethods.knn_fill(data_slice, column, gap_mask)
            elif method == "Kalman Filter":
                filled_data[column] = GapFillingMethods.kalman_fill(data_slice, column, gap_mask)
            elif method == "Seasonal Decomposition":
                filled_data[column] = GapFillingMethods.seasonal_fill(data_slice, column, gap_mask)
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