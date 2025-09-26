%% COVID-19 Analysis Runner
%
% This script sets up the path and runs the main COVID-19 analysis.
% Run this file from the project root directory.

% Add source directory to path
addpath(genpath('src'));
addpath(genpath('data'));

% Ensure output directories exist
if ~exist('figures', 'dir')
    mkdir('figures');
end
if ~exist('data', 'dir')
    mkdir('data');
end

% Run the main analysis
fprintf('Starting COVID-19 Time Series Analysis...\n');
fprintf('========================================\n\n');

MIDA_project_COVID19;

fprintf('\nAnalysis completed successfully!\n');
fprintf('Check the figures/ directory for output plots.\n');