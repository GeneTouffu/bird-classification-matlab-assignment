function img_ids = get_img_ids(file)
    fid = fopen(file, 'r');
    lines = textscan(fid, '%d %s');

    img_ids = lines{1};
 end
