%% COVID-19 Time Series Analysis using AR and ARMA Models
%
% This script analyzes Italian COVID-19 data using autoregressive (AR) and
% autoregressive moving average (ARMA) models to understand the dynamics
% across different pandemic periods.
%
% Author: Lorenzo
% Date: 2024

clear; clc; close all;

%% Configuration
MAX_AR_ORDER = 6;
MAX_MA_ORDER = 6;
TRAIN_RATIO = 0.7;

%% 1. Data Loading and Preprocessing
fprintf('Loading COVID-19 data from Italian Civil Protection...\n');
url = 'https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv';
data = readtable(url);

% Extract and preprocess new positive cases
y_raw = data.nuovi_positivi;
y_raw(y_raw < 0) = 0;  % Remove negative values

% Parse timestamps
timestamps = datetime(data.data, 'InputFormat','yyyy-MM-dd''T''HH:mm:ss');

% Normalize data (z-score)
mu_global = mean(y_raw);
sigma_global = std(y_raw);
y_normalized = (y_raw - mu_global) / sigma_global;

%% 2. Define Analysis Periods
period_definitions = [
    datetime(2020,3,9),   datetime(2020,5,3);    % Total lockdown
    datetime(2020,5,4),   datetime(2020,11,5);   % Reopening phase
    datetime(2021,3,8),   datetime(2021,9,8);    % Color zones system
    datetime(2020,12,27), datetime(2021,6,30)    % Vaccination campaign
];

period_names = {'Lockdown totale', 'Riaperture', 'Zone colori', 'Campagna vax'};

% Create period indices
periods = cell(size(period_definitions, 1), 1);
for i = 1:size(period_definitions, 1)
    period_mask = timestamps >= period_definitions(i,1) & timestamps <= period_definitions(i,2);
    periods{i} = find(period_mask);
end

%% 3. Initialize Results Storage
num_periods = length(periods);
SSR_results_AR = nan(MAX_AR_ORDER, num_periods);
SSR_results_ARMA = nan(MAX_AR_ORDER, MAX_MA_ORDER, num_periods);
R2_results_AR = nan(MAX_AR_ORDER, num_periods);
R2_results_ARMA = nan(MAX_AR_ORDER, MAX_MA_ORDER, num_periods);

% Store optimal model orders
optimal_AR_orders = zeros(num_periods, 1);
optimal_ARMA_orders = zeros(num_periods, 2);

%% 4. Model Training and Validation
fprintf('\nStarting model analysis across %d periods...\n\n', num_periods);

for period_idx = 1:num_periods
    fprintf('Processing period %d: %s\n', period_idx, period_names{period_idx});

    % Extract period data
    period_indices = periods{period_idx};
    y_period = y_normalized(period_indices);
    n_samples = length(y_period);

    % Split into training and validation sets
    n_train = round(TRAIN_RATIO * n_samples);
    y_train = y_period(1:n_train);
    y_validation = y_period(n_train+1:end);
    y_val_iddata = iddata(y_validation, [], 1);

    fprintf('  - Training samples: %d, Validation samples: %d\n', n_train, length(y_validation));

    %% 4.1 AR Model Analysis
    fprintf('  - Evaluating AR models (orders 1-%d)...\n', MAX_AR_ORDER);
    for ar_order = 1:MAX_AR_ORDER
        try
            ar_model = ar(y_train, ar_order, 'ls');
            y_pred_ar = predict(ar_model, y_val_iddata, 1);
            prediction_error = y_validation - y_pred_ar.y;

            SSR_results_AR(ar_order, period_idx) = sum(prediction_error.^2);
            R2_results_AR(ar_order, period_idx) = calculateR2(y_validation, y_pred_ar.y);
        catch ME
            fprintf('    Warning: AR(%d) failed - %s\n', ar_order, ME.message);
        end
    end

    %% 4.2 ARMA Model Analysis
    fprintf('  - Evaluating ARMA models...\n');
    for ar_order = 1:MAX_AR_ORDER
        for ma_order = 1:MAX_MA_ORDER
            try
                arma_model = armax(y_train, [ar_order ma_order]);
                y_pred_arma = predict(arma_model, y_val_iddata, 1);
                prediction_error = y_validation - y_pred_arma.y;

                SSR_results_ARMA(ar_order, ma_order, period_idx) = sum(prediction_error.^2);
                R2_results_ARMA(ar_order, ma_order, period_idx) = calculateR2(y_validation, y_pred_arma.y);
            catch ME
                fprintf('    Warning: ARMA(%d,%d) failed - %s\n', ar_order, ma_order, ME.message);
            end
        end
    end

    %% 4.3 Find Optimal Models
    [~, best_ar_order] = min(SSR_results_AR(:, period_idx));
    [min_ssr_arma, linear_idx] = min(SSR_results_ARMA(:,:,period_idx), [], 'all', 'omitnan');
    [best_arma_ar, best_arma_ma] = ind2sub([MAX_AR_ORDER, MAX_MA_ORDER], linear_idx);

    optimal_AR_orders(period_idx) = best_ar_order;
    optimal_ARMA_orders(period_idx, :) = [best_arma_ar, best_arma_ma];

    %% 4.4 Display Results
    fprintf('\n=== RESULTS FOR %s ===\n', upper(period_names{period_idx}));
    fprintf('Optimal AR model: AR(%d) - SSR=%.5f, R²=%.3f\n', ...
        best_ar_order, SSR_results_AR(best_ar_order, period_idx), R2_results_AR(best_ar_order, period_idx));
    fprintf('Optimal ARMA model: ARMA(%d,%d) - SSR=%.5f, R²=%.3f\n', ...
        best_arma_ar, best_arma_ma, min_ssr_arma, R2_results_ARMA(best_arma_ar, best_arma_ma, period_idx));

    %% 4.5 Generate Validation Plots
    generateValidationPlot(y_train, y_validation, best_ar_order, [best_arma_ar, best_arma_ma], ...
                          period_names{period_idx}, n_train);
end

%% 5. Summary Visualization
generateSummaryPlots(SSR_results_AR, SSR_results_ARMA, period_names, MAX_AR_ORDER, MAX_MA_ORDER);

fprintf('\n=== ANALYSIS COMPLETE ===\n');
fprintf('Summary of optimal model orders:\n');
for i = 1:num_periods
    fprintf('  %s: AR(%d), ARMA(%d,%d)\n', period_names{i}, ...
            optimal_AR_orders(i), optimal_ARMA_orders(i,1), optimal_ARMA_orders(i,2));
end

%% Helper Functions
function r2 = calculateR2(y_actual, y_predicted)
    % Calculate R-squared coefficient
    ss_res = sum((y_actual - y_predicted).^2);
    ss_tot = sum((y_actual - mean(y_actual)).^2);
    r2 = 1 - (ss_res / ss_tot);
end

function generateValidationPlot(y_train, y_validation, best_ar, best_arma, period_name, n_train)
    % Generate validation plots for best models
    y_val_iddata = iddata(y_validation, [], 1);

    % Fit best models
    ar_model = ar(y_train, best_ar, 'ls');
    y_ar_pred = predict(ar_model, y_val_iddata, 1);

    arma_model = armax(y_train, [best_arma(1) best_arma(2)]);
    y_arma_pred = predict(arma_model, y_val_iddata, 1);

    % Create plot
    figure('Name', sprintf('Validation Results - %s', period_name));
    validation_indices = n_train+1:n_train+length(y_validation);

    plot(validation_indices, y_validation, 'ko', 'MarkerFaceColor','k', 'MarkerSize',5, 'DisplayName','Actual');
    hold on;
    plot(validation_indices, y_ar_pred.y, 'bo', 'MarkerFaceColor','b', 'MarkerSize',4, 'DisplayName',sprintf('AR(%d)',best_ar));
    plot(validation_indices, y_arma_pred.y, 'r^', 'MarkerFaceColor','r', 'MarkerSize',4, 'DisplayName',sprintf('ARMA(%d,%d)',best_arma(1),best_arma(2)));

    title(sprintf('%s - Model Validation\nBest AR(%d) vs Best ARMA(%d,%d)', period_name, best_ar, best_arma(1), best_arma(2)));
    xlabel('Time Index');
    ylabel('Normalized New Cases');
    legend('Location','best');
    grid on;
    hold off;
end

function generateSummaryPlots(SSR_AR, SSR_ARMA, period_names, max_ar, max_ma)
    % Generate comprehensive summary plots
    figure('Position', [100 100 1400 800], 'Name', 'Model Performance Summary');

    for p = 1:length(period_names)
        subplot(2,2,p);
        hold on;

        % Plot AR results
        plot(1:max_ar, SSR_AR(:,p), 'ko-', 'LineWidth',2, 'MarkerFaceColor','k', 'DisplayName','AR Models');

        % Plot ARMA results for different MA orders
        colors = lines(max_ma);
        for ma_order = 1:max_ma
            plot(1:max_ar, SSR_ARMA(:,ma_order,p), 'o-', 'Color',colors(ma_order,:), ...
                'MarkerFaceColor',colors(ma_order,:), 'DisplayName',sprintf('ARMA (MA=%d)',ma_order));
        end

        % Highlight best ARMA model
        [min_ssr, linear_idx] = min(SSR_ARMA(:,:,p), [], 'all', 'omitnan');
        [best_ar, best_ma] = ind2sub([max_ar, max_ma], linear_idx);
        plot(best_ar, min_ssr, 'gp', 'MarkerSize',14, 'LineWidth',2, 'DisplayName','Best ARMA');

        title(period_names{p}, 'Interpreter','none');
        xlabel('AR Order');
        ylabel('Validation SSR');
        legend('show', 'Location','northeastoutside');
        grid on;
        hold off;
    end

    sgtitle('Model Performance Comparison Across COVID-19 Periods', 'FontSize', 14, 'FontWeight', 'bold');
end

