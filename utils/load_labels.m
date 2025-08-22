function labels = load_labels(filepath)
    fid = fopen(filepath, 'r');
    data = textscan(fid, '%d %d');
    fclose(fid);
    labels = containers.Map(data{1}, data{2});
end
