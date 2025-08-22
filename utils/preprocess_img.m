function img_out = preprocess_img(img)
    targetSize = [256, 256];
    img_resized = imresize(img, targetSize);

    if size(img_resized, 3) == 3
        img_gray = rgb2gray(img_resized);
    else
        img_gray = img_resized;
    end

    img_norm = im2double(img_gray);

    img_out = img_norm;
end
