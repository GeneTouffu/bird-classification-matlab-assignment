addpath(genpath('scripts'));

% Create a directory for results if it doesn't exist
if ~exist('results', 'dir')
    mkdir('results');
end

% Run experiments and save results
experiments = {'1', '2', '3', '4'};
for i = 1:length(experiments)
    exp_num = experiments{i};
    fprintf("Running Experiment %s\n", exp_num);
    
    % Run the experiment
    results = eval(sprintf('run_experiment_%s()', exp_num));
    
    % Save confusion matrix
    confusion_filename = sprintf('results/exp%s_confusion_matrix.xlsx', exp_num);
    writematrix(results.confusion_matrix, confusion_filename);
    
    % Save class statistics
    stats_filename = sprintf('results/exp%s_class_stats.xlsx', exp_num);
    writetable(results.class_stats_table, stats_filename);
    
    % Save accuracy to a text file
    accuracy_filename = sprintf('results/exp%s_accuracy.txt', exp_num);
    fid = fopen(accuracy_filename, 'w');
    fprintf(fid, 'Weighted Accuracy: %.2f%%', results.accuracy);
    fclose(fid);
    
    fprintf("Saved results for Experiment %s\n\n", exp_num);
end