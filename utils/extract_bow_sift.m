function [X, y] = extract_bow_sift(img_ids, label_map, vocabulary, x)
    X = [];
    y = zeros(length(img_ids), 1);
    numClusters = size(vocabulary, 2);

    for i = 1:length(img_ids)
        img = read_img(img_ids(i), x);
        img = preprocess_img(img);
        points = detectSIFTFeatures(img);
        [features, ~] = extractFeatures(img, points);

        if isempty(features)
            hist_vec = zeros(1, numClusters);
        else
            % Assign descriptors to nearest cluster centers
            dists = pdist2(double(features), double(vocabulary'));
            [~, idx] = min(dists, [], 2);
            hist_vec = histcounts(idx, 1:(numClusters+1));
            hist_vec = hist_vec / norm(hist_vec);  % normalize
        end

        X = [X; hist_vec];
        y(i) = label_map(img_ids(i));
    end
end
