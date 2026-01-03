addpath('./functions');   % helper funcs

input_dir = fullfile('..','frames_raw');
output_cs = fullfile('..','frames_processed');      % TwIST output
output_bp = fullfile('..','frames_processed_bp');   % BP output

% Make sure output folders exist
if ~isfolder(output_cs), mkdir(output_cs); end
if ~isfolder(output_bp), mkdir(output_bp); end

% Collect images
exts = {'.png','.jpg','.jpeg','.tif','.tiff','.bmp'};
files = [];
for i = 1:numel(exts)
    files = [files; dir(fullfile(input_dir, ['*' exts{i}]))]; %#ok<AGROW>
    files = [files; dir(fullfile(input_dir, ['*' upper(exts{i})]))]; %#ok<AGROW>
end
if isempty(files)
    fprintf('No images found in %s\n', input_dir);
    return;
end
[~, ia] = unique(fullfile({files.folder}, {files.name}), 'stable');
files = files(ia);

% Process each image
for k = 1:numel(files)
    in_path = fullfile(files(k).folder, files(k).name);
    fprintf('Processing %s ...\n', files(k).name);
    I = imread(in_path);
%{
    {tried handling rgb channels seperately}

    try
        [bp, cs] = lensless_reconstruct(I);  % your function (unchanged)
    catch ME
        fprintf('  Skipping %s: %s\n', files(k).name, ME.message);
        continue;
    end

    % Save with EXACT same filename (name + extension)
    imwrite(cs, fullfile(output_cs, files(k).name));
    imwrite(bp, fullfile(output_bp, files(k).name));
%}
    try
        [bp, cs] = lensless_reconstruct(I);  % your function (unchanged)
    catch ME
        fprintf('  Skipping %s: %s\n', files(k).name, ME.message);
        continue;
    end

    bp = center_crop(bp, 215, 300);
    cs = center_crop(cs, 215, 300);

    % Save with EXACT same filename (name + extension)
    imwrite(mat2gray(cs), fullfile(output_cs, files(k).name));
    imwrite(mat2gray(bp), fullfile(output_bp, files(k).name));

end

fprintf('Done. Saved CS to "%s" and BP to "%s".\n', output_cs, output_bp);
