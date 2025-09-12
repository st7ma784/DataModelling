# Solar Wind Time Series Gap Filling Interface

A web application that demonstrates different methods for filling gaps in solar wind time series data.

## Features

- **Synthetic Data Generation**: Creates realistic solar wind data using multiple cosine waves + noise
- **Gap Simulation**: Introduces artificial gaps (up to 1 hour) to simulate missing data
- **Multiple Gap-Filling Methods**:
  - Linear Interpolation
  - FFT Reconstruction
  - LSTM Neural Networks
  - Transformer Models
  - Bayesian Modeling with Gumbel Softmax
- **Interactive Visualization**: 2D time series and 3D vector plots
- **Real-time Comparison**: Compare original, gapped, and filled data

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
streamlit run solar_wind_app.py
```

## Usage

1. Select a time range using the slider (0-168 hours)
2. Choose a gap-filling method from the dropdown
3. View the results in the plots:
   - Left panel: B vector (magnetic field) components
   - Right panel: V vector (velocity) components
   - Bottom: 3D vector visualizations

## Methods

- **Linear Interpolation**: Simple linear interpolation between gap boundaries
- **FFT Reconstruction**: Uses Fourier analysis of surrounding data to reconstruct gaps
- **LSTM Prediction**: Recurrent neural network trained on non-gap data
- **Transformer**: Attention-based sequence model for gap prediction
- **Bayesian + Gumbel Softmax**: Probabilistic modeling with smooth sampling

## Data

The synthetic solar wind data includes:
- B vector (magnetic field): Bx, By, Bz components
- V vector (velocity): Vx, Vy, Vz components
- Generated using 5 different cosine frequencies
- Gaussian noise added for realism
- 1-minute sampling resolution starting from 1981