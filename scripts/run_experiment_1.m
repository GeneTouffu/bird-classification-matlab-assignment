function results = run_experiment_1()
    addpath(genpath('utils'));
    addpath(genpath('models'));
    train_img_ids = get_img_ids("train.txt");
    validate_img_ids = get_img_ids("validate.txt");
    test_img_ids = get_img_ids("test.txt");
    labels = load_labels("image_class_labels.txt");

    bow = build_vocabulary(train_img_ids, labels, 100, 0);

    [X_train, y_train] = extract_bow_sift(train_img_ids, labels, bow, 0);
    [X_val, y_val]     = extract_bow_sift(validate_img_ids, labels, bow, 0);
    [X_test, y_test]   = extract_bow_sift(test_img_ids, labels, bow, 0);

    mu = mean(X_train);
    sigma = std(X_train);
    X_train_norm = (X_train - mu) ./ sigma;
    X_val_norm   = (X_val - mu) ./ sigma;
    X_test_norm  = (X_test - mu) ./ sigma;

    [coeff, score_train, ~, ~, explained] = pca(X_train_norm);
    cumulative_variance = cumsum(explained);
    n_components = find(cumulative_variance >= 95, 1);
    fprintf("PCA reducing to %d components (95%% variance)\n", n_components);

    % Project validation and test sets
    score_val = (X_val_norm - mean(X_train_norm)) * coeff(:, 1:n_components);
    score_test = (X_test_norm - mean(X_train_norm)) * coeff(:, 1:n_components);

    template = templateSVM('KernelFunction', 'linear', 'Standardize', false);
    model = fitcecoc(score_train(:, 1:n_components), y_train, 'Learners', template);

    % --- Evaluate on test set ---
    y_pred = predict(model, score_test);
    
    % Calculate overall accuracy
    test_acc = sum(y_pred == y_test) / length(y_test);
    fprintf("Test Accuracy: %.2f%%\n", test_acc * 100);

    % Confusion matrix
    [conf_mat, class_labels] = confusionmat(y_test, y_pred);

    % Per-class performance
    num_classes = length(class_labels);
    class_stats = zeros(num_classes, 3); % [Class ID, Correct Rate, Incorrect Rate]
    total_samples = sum(conf_mat(:));
    correct_sum = 0;

    for i = 1:num_classes
        class_id = double(class_labels(i));
        total_in_class = sum(conf_mat(i, :));
        correct_in_class = conf_mat(i, i);
        correct_rate = correct_in_class / total_in_class;
        incorrect_rate = 1 - correct_rate;

        class_stats(i, :) = [class_id, correct_rate * 100, incorrect_rate * 100];

        correct_sum = correct_sum + correct_in_class;
    end

    % Weighted average accuracy
    weighted_accuracy = correct_sum / total_samples * 100;

    % --- Plot confusion matrix ---
    figure;
    confusionchart(y_test, y_pred);
    title("Experiment 1");

    % --- Store results ---
    results = struct( ...
        'model', model, ...                          % The trained model object
        'accuracy', weighted_accuracy, ...           % Weighted average accuracy
        'y_pred', y_pred, ...                        % Predicted labels
        'y_test', y_test, ...                        % True labels
        'confusion_matrix', conf_mat, ...            % Confusion matrix
        'class_labels', class_labels, ...            % Class labels
        'class_stats_table', array2table(class_stats, ... 
            'VariableNames', {'ClassID', 'CorrectRate', 'IncorrectRate'}) ...  % Per-class correct/incorrect rates
    );
end
