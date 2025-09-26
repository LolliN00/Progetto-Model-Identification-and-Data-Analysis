# COVID-19 Time Series Analysis with AR/ARMA Models

A comprehensive MATLAB analysis of Italian COVID-19 data using autoregressive (AR) and autoregressive moving average (ARMA) models to understand pandemic dynamics across different periods.

##  Overview

This project analyzes the temporal patterns of COVID-19 new cases in Italy using time series modeling techniques. The analysis focuses on four distinct periods of the pandemic, each characterized by different policy interventions and social dynamics.

##  Objectives

- **Model Selection**: Compare AR and ARMA models of different orders to find optimal representations
- **Period Analysis**: Analyze four key periods of the Italian COVID-19 pandemic
- **Validation**: Use train/validation splits to ensure robust model performance
- **Visualization**: Generate comprehensive plots for model comparison and results interpretation

##  Analysis Periods

1. **Lockdown totale** (March 9, 2020 - May 3, 2020): Total lockdown period
2. **Riaperture** (May 4, 2020 - November 5, 2020): Reopening phase
3. **Zone colori** (March 8, 2021 - September 8, 2021): Color zones system
4. **Campagna vax** (December 27, 2020 - June 30, 2021): Vaccination campaign

##  Requirements

### MATLAB Toolboxes
- System Identification Toolbox
- Signal Processing Toolbox
- Statistics and Machine Learning Toolbox

### MATLAB Version
- MATLAB R2019b or later recommended

##  Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/covid19-timeseries-analysis.git
   cd covid19-timeseries-analysis
   ```

2. **Run the main analysis**:
   ```matlab
   run_analysis  % Recommended - sets up paths automatically
   ```

   Or directly:
   ```matlab
   cd src
   MIDA_project_COVID19
   ```

3. **View results**: The script will:
   - Download latest COVID-19 data automatically
   - Process and normalize the data
   - Fit AR and ARMA models for each period
   - Generate validation plots
   - Display optimal model orders

## 📈 Methodology

### Data Preprocessing
- **Source**: Italian Civil Protection COVID-19 dataset
- **Normalization**: Z-score standardization
- **Filtering**: Removal of negative values

### Model Training
- **Train/Validation Split**: 70%/30% ratio
- **Model Orders**: AR(1-6), ARMA(1-6, 1-6)
- **Estimation Method**: Least Squares (LS) for AR, Maximum Likelihood for ARMA
- **Performance Metrics**: Sum of Squared Residuals (SSR), R-squared

### Validation Strategy
- One-step-ahead prediction on validation set
- Cross-period comparison of optimal model orders
- Visual inspection of prediction accuracy

##  Project Structure

```
covid19-timeseries-analysis/
├── src/
│   ├── MIDA_project_COVID19.m      # Main analysis script
│   └── run_analysis.m              # Runner script (recommended entry point)
├── data/
│   └── dpc-covid19-ita-andamento-nazionale.csv  # Data file
├── figures/
│   └── results.fig                 # Generated plots and figures
├── docs/                          # Documentation directory
├── resources/                     # MATLAB project resources
├── README.md                      # This file
├── LICENSE                        # MIT License
└── .gitignore                    # Git ignore rules
```

## Output

### Console Output
- Training/validation sample counts for each period
- Model fitting progress and warnings
- Optimal model orders with performance metrics
- Summary table of best models across all periods

### Visualizations
- **Validation Plots**: Comparison of actual vs predicted values for optimal models
- **Performance Summary**: SSR comparison across different model orders
- **Period Comparison**: Side-by-side analysis of all four periods
