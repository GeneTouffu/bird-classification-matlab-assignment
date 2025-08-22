function [X, y] = extract_avg_sift(img_ids, label_map, x)
    X = [];
    y = zeros(length(img_ids), 1);
    for i = 1:length(img_ids)
        img = read_img(img_ids(i), x);
        img = preprocess_img(img);

        % Extract SIFT feature
        points = detectSIFTFeatures(img);
        
        [features, ~] = extractFeatures(img, points);

        % Average descriptor vectors to create fixed-length feature vector
        if isempty(features)
            avg_feat = zeros(1, 128);  % fallback if no features
        else
            avg_feat = mean(features, 1);
        end

        X = [X; avg_feat];
        y(i) = label_map(img_ids(i));
    end
end
