function cropped = center_crop(img, cropH, cropW)
%CENTER_CROP  Crop a region of size cropH x cropW from the center of img
%
%   cropped = center_crop(img, cropH, cropW)
%
%   Inputs:
%       img    : input 2D (grayscale) or 3D (RGB) image
%       cropH  : desired crop height
%       cropW  : desired crop width
%
%   Output:
%       cropped : cropped image of size [cropH, cropW, (channels)]

    [H, W, C] = size(img);

    % Find center
    centerY = floor(H/2);
    centerX = floor(W/2);

    % Compute crop bounds
    r1 = centerY - floor(cropH/2) + 1;
    r2 = r1 + cropH - 1;
    c1 = centerX - floor(cropW/2) + 1;
    c2 = c1 + cropW - 1;

    % Ensure indices are valid
    if r1 < 1 || c1 < 1 || r2 > H || c2 > W
        error('Crop window [%d x %d] is too large for image size [%d x %d].', ...
              cropH, cropW, H, W);
    end

    % Crop
    cropped = img(r1:r2, c1:c2, :);
end
