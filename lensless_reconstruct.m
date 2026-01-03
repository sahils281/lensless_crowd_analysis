function [bp_out, cs_out] = lensless_reconstruct(img)
% LENSLESS_RECONSTRUCT
%   Perform FZA lensless imaging reconstruction on a given image.
%   Input:
%       img : grayscale image (HxW), numeric array
%   Output:
%       bp_out : back-propagation reconstructed image
%       cs_out : compressed-sensing (TwIST) reconstructed image
%
% Requirements: functions/ folder must contain:
%   pinhole.m, FZA.m, MyForwardOperatorPropagation.m,
%   MyAdjointOperatorPropagation.m, TwIST.m, tvdenoise.m, TVnorm.m

    addpath('./functions');

    % --- Basic checks ---
    if ~isnumeric(img)
        error('Input must be a numeric array (grayscale).');
    end
    img = im2double(img);

    % --- Parameters (fixed) ---
    di = 3;          % distance from mask to sensor
    z1 = 20;  x1 = 0;  y1 = 0;
    Lx1 = 20;        % object size
    dp  = 0.01;      % pixel pitch
    Nx  = 512; Ny = 512;
    S   = 2*dp*Nx;   % aperture diameter
    r1  = 0.23;      % FZA constant
    M   = di/z1;
    ri  = (1+M)*r1;

    % --- Imaging ---
    Im = pinhole(img, di, x1, y1, z1, Lx1, dp, Nx);

    mask = FZA(S, 2*Nx, ri);
    I = conv2(Im, mask, 'same') * 2*dp*dp/ri^2;
    I = I - mean(I(:));

    % --- Fresnel transfer function ---
    fu_max = 0.5 / dp;
    fv_max = 0.5 / dp;
    du = 2*fu_max / Nx;
    dv = 2*fv_max / Ny;
    [u,v] = meshgrid(-fu_max:du:fu_max-du, -fv_max:dv:fv_max-dv);
    H = 1i * exp(-1i * (pi*ri^2) * (u.^2 + v.^2));

    % --- Back-propagation ---
    Or = MyAdjointOperatorPropagation(I, H);
    bp_out = real(Or);

    % --- Propagation operators ---
    A  = @(obj) MyForwardOperatorPropagation(obj, H);
    AT = @(Iin) MyAdjointOperatorPropagation(Iin, H);

    % --- TwIST settings ---
    tv_iters = 2;
    Psi = @(x,th) tvdenoise(x, 2/th, tv_iters);
    Phi = @(x) TVnorm(x);
    tau = 0.005; tolA = 1e-6; iterations = 200;

    % --- TwIST reconstruction ---
    [f_reconstruct,~,~,~,~,~] = TwIST(I, A, tau, ...
                                'AT', AT, ...
                                'Psi', Psi, ...
                                'Phi', Phi, ...
                                'Initialization', 2, ...
                                'Monotone', 1, ...
                                'StopCriterion', 1, ...
                                'MaxIterA', iterations, ...
                                'MinIterA', iterations, ...
                                'ToleranceA', tolA, ...
                                'Verbose', 1);
    cs_out = real(f_reconstruct);

    % --- (Plots commented out, but left in case needed) ---
    % figure, imagesc(Im); title('Original image'); colormap gray; axis image off
    % figure, imagesc(mask); title('FZA pattern'); colormap gray; axis image off
    % figure, imagesc(I); title('Observed imaging'); colormap gray; axis image off
    % figure, imagesc(bp_out); title('Reconstructed image (BP)'); colormap gray; axis image off
    % figure, imagesc(cs_out); title('Reconstructed image (CS)'); colormap gray; axis image off
end
