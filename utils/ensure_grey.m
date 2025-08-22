function img_out = ensure_grey(filename)
    [img, map] = imread(filename);
    if ~isempty(map)
        img = ind2rgb(img, map);
        img = im2uint8(img);
    end
    if size(img, 3) == 1
        img_out = img;
    else
        img_out = rgb2gray(img);
    end
end