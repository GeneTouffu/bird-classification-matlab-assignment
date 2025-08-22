function vocabulary = build_vocabulary(img_ids, label_map, numClusters, x)
    descriptors = [];

    for i = 1:length(img_ids)
        img = read_img(img_ids(i), x);
        img = preprocess_img(img);
        points = detectSIFTFeatures(img);
        [features, ~] = extractFeatures(img, points);

        if ~isempty(features)
            descriptors = [descriptors; double(features)];
        end
    end

    % K-means clustering to build vocabulary
    [~, vocabulary] = kmeans(descriptors, numClusters, 'MaxIter', 500);
    vocabulary = vocabulary';  % 128 x numClusters
end
