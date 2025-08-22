function results = run_experiment_4()
    addpath(genpath('utils'));

    % --- Load image IDs and original labels ---
    train_img_ids = get_img_ids("train.txt");
    validate_img_ids = get_img_ids("validate.txt");
    test_img_ids = get_img_ids("test.txt");
    labels = load_labels("image_class_labels.txt");

    % Convert image IDs to cell arrays
    train_img_ids_cell = num2cell(train_img_ids);
    validate_img_ids_cell = num2cell(validate_img_ids);
    test_img_ids_cell = num2cell(test_img_ids);

    % --- Extract raw numeric labels ---
    train_labels_raw = cell2mat(values(labels, train_img_ids_cell));
    val_labels_raw   = cell2mat(values(labels, validate_img_ids_cell));
    test_labels_raw  = cell2mat(values(labels, test_img_ids_cell));

    % --- Remap labels to 1:N based on training set only ---
    unique_labels = unique(train_labels_raw);
    label_map = containers.Map(unique_labels, 1:length(unique_labels));

    train_labels = categorical(cell2mat(values(label_map, num2cell(train_labels_raw))));
    val_labels   = categorical(cell2mat(values(label_map, num2cell(val_labels_raw))));
    test_labels  = categorical(cell2mat(values(label_map, num2cell(test_labels_raw))));

    num_classes = numel(unique(train_labels));

    % --- Create directory to save preprocessed images ---
    if ~exist('saved_images', 'dir')
        mkdir('saved_images');
    end

    % --- Save and preprocess images ---
    box_mode = 1;
    train_imgs_paths = cell(length(train_img_ids), 1);
    val_imgs_paths   = cell(length(validate_img_ids), 1);
    test_imgs_paths  = cell(length(test_img_ids), 1);

    for i = 1:length(train_img_ids)
        img = read_img(train_img_ids(i), box_mode);
        img = preprocess_img(img);
        img_path = sprintf('saved_images/4train_%d.png', train_img_ids(i));
        imwrite(img, img_path);
        train_imgs_paths{i} = img_path;
    end

    for i = 1:length(validate_img_ids)
        img = read_img(validate_img_ids(i), box_mode);
        img = preprocess_img(img);
        img_path = sprintf('saved_images/4val_%d.png', validate_img_ids(i));
        imwrite(img, img_path);
        val_imgs_paths{i} = img_path;
    end

    for i = 1:length(test_img_ids)
        img = read_img(test_img_ids(i), box_mode);
        img = preprocess_img(img);
        img_path = sprintf('saved_images/4test_%d.png', test_img_ids(i));
        imwrite(img, img_path);
        test_imgs_paths{i} = img_path;
    end

    % --- ImageDatastores with grayscale-safe loading ---
    imdsTrain = imageDatastore(train_imgs_paths, ...
        'Labels', train_labels, ...
        'ReadFcn', @(filename) imresize(ensure_grey(filename), [256, 256]));
    imdsVal = imageDatastore(val_imgs_paths, ...
        'Labels', val_labels, ...
        'ReadFcn', @(filename) imresize(ensure_grey(filename), [256, 256]));
    imdsTest = imageDatastore(test_imgs_paths, ...
        'Labels', test_labels, ...
        'ReadFcn', @(filename) imresize(ensure_grey(filename), [256, 256]));

    % --- Define CNN architecture for N classes ---
    layers = [
        imageInputLayer([256 256 1], 'Name', 'input')

        convolution2dLayer(3, 32, 'Padding', 'same')
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer(2, 'Stride', 2)

        convolution2dLayer(3, 64, 'Padding', 'same')
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer(2, 'Stride', 2)

        fullyConnectedLayer(num_classes, 'Name', 'fc')
        softmaxLayer
        classificationLayer
    ];

    % --- Set training options ---
    options = trainingOptions('sgdm', ...
        'InitialLearnRate', 0.001, ...
        'MaxEpochs', 5, ...
        'MiniBatchSize', 35, ...
        'ValidationData', imdsVal, ...
        'Shuffle', 'every-epoch', ...
        'Plots','training-progress');

    % --- Train the network ---
    trained_net = trainNetwork(imdsTrain, layers, options);

    % --- Predict and evaluate ---
    y_pred = classify(trained_net, imdsTest);
    test_acc = sum(y_pred == imdsTest.Labels) / numel(imdsTest.Labels);
    fprintf("Test Accuracy: %.2f%%\n", test_acc * 100);

    [conf_mat, class_labels] = confusionmat(imdsTest.Labels, y_pred);

    % --- Per-class stats ---
    num_classes = length(class_labels);
    class_stats = zeros(num_classes, 3);
    total_samples = sum(conf_mat(:));
    correct_sum = 0;

    for i = 1:num_classes
        class_id = double(class_labels(i));
        total_in_class = sum(conf_mat(i, :));
        correct_in_class = conf_mat(i, i);
        correct_rate = correct_in_class / max(1, total_in_class);
        incorrect_rate = 1 - correct_rate;
        class_stats(i, :) = [class_id, correct_rate * 100, incorrect_rate * 100];
        correct_sum = correct_sum + correct_in_class;
    end

    weighted_accuracy = correct_sum / total_samples * 100;

    % --- Confusion matrix ---
    figure;
    confusionchart(imdsTest.Labels, y_pred, 'RowSummary','row-normalized');
    title("Experiment 4 - CNN Confusion Matrix");

    % --- Store results ---
    results = struct( ...
        'model', trained_net, ...
        'accuracy', weighted_accuracy, ...
        'y_pred', y_pred, ...
        'y_test', imdsTest.Labels, ...
        'confusion_matrix', conf_mat, ...
        'class_labels', class_labels, ...
        'class_stats_table', array2table(class_stats, ...
            'VariableNames', {'ClassID', 'CorrectRate', 'IncorrectRate'}) ...
    );
end
