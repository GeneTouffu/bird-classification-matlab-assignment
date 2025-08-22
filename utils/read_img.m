function img_cropped = read_img(image_id, box_mode)
    images_txt_path = "images.txt";
    bbox_txt_path = "bounding_boxes.txt";
    dataset_root = "images";
    fid = fopen(images_txt_path, 'r');
    lines = textscan(fid, '%d %s');
    fclose(fid);

    rel_path = lines{2}{image_id};
    img_path = fullfile(dataset_root, rel_path);
    img = imread(img_path);

    if box_mode == 1
        % --- Read bounding box data ---
        bbox_data = dlmread(bbox_txt_path);
        bbox = bbox_data(bbox_data(:, 1) == image_id, :);
        if isempty(bbox)
            error('Bounding box not found for image ID %d', image_id);
        end

        x = round(bbox(2));
        y = round(bbox(3));
        w = round(bbox(4));
        h = round(bbox(5));

        % --- Make sure crop area is within image bounds ---
        [imgH, imgW, ~] = size(img);
        x = max(1, x);
        y = max(1, y);
        x_end = min(x + w - 1, imgW);
        y_end = min(y + h - 1, imgH);

        % --- Crop image ---
        img_cropped = img(y:y_end, x:x_end, :);
    else
        img_cropped = img;
    end

end

