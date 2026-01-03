function [bp_out, cs_out] = lensless_reconstruct_bp_cs(img)
% LENSLESS_RECONSTRUCT
%   Perform FZA lensless imaging reconstruction.
%   If input is RGB (HxWx3), each channel is processed independently with
%   the exact same logic as your grayscale script, then stacked back.
%   If input is grayscale (HxW), it runs once (unchanged math).
%
%   Inputs:
%       img : grayscale (HxW) or RGB (HxWx3), numeric
%   Outputs:
%       bp_out : back-propagation reconstruction (grayscale or RGB)
%       cs_out : compressed-sensing (TwIST) reconstruction (grayscale or RGB)
%
% Requirements: functions/ folder must contain:
%   pinhole.m, FZA.m, MyForwardOperatorPropagation.m,
%   MyAdjointOperatorPropagation.m, TwIST.m, tvdenoise.m, TVnorm.m

    addpath('./functions');

    if ~isnumeric(img)
        error('Input must be a numeric array (grayscale or RGB).');
    end
    img = im2double(img);

    if ndims(img) == 3 && size(img,3) == 3
        % --- RGB: process each channel with identical logic and stack ---
        [H,W,~] = size(img);
        bp_out = zeros(512,512,3);
        cs_out = zeros(512,512,3);
        for k = 1:3
            [bp_out(:,:,k), cs_out(:,:,k)] = recon_single_channel(img(:,:,k));
        end
    else
        % --- Grayscale: run once (unchanged logic) ---
        [bp_out, cs_out] = recon_single_channel(img);
    end

    % ===== Nested helper: exact single-channel logic (unchanged math) =====
    function [bp_ch, cs_ch] = recon_single_channel(img2d)
        % Alias to keep original variable name 'img' for unchanged lines
        img = img2d;

        % --- Parameters (fixed, identical to your script) ---
        di = 3;          % distance from mask to sensor
        z1 = 20;  x1 = 0;  y1 = 0;
        Lx1 = 20;        % object size
        dp  = 0.01;      % pixel pitch
        Nx  = 512; Ny = 512;
        S   = 2*dp*Nx;   % aperture diameter
        r1  = 0.23;      % FZA constant
        M   = di/z1;
        ri  = (1+M)*r1;

        % --- Imaging (unchanged) ---
        Im = pinhole(img, di, x1, y1, z1, Lx1, dp, Nx);

        mask = FZA(S, 2*Nx, ri);
        I = conv2(Im, mask, 'same') * 2*dp*dp/ri^2;  % scale to match I1
        I = I - mean(I(:));

        % --- Fresnel transfer function (unchanged) ---
        fu_max = 0.5 / dp;
        fv_max = 0.5 / dp;
        du = 2*fu_max / Nx;
        dv = 2*fv_max / Ny;
        [u,v] = meshgrid(-fu_max:du:fu_max-du, -fv_max:dv:fv_max-dv);
        H = 1i * exp(-1i * (pi*ri^2) * (u.^2 + v.^2));

        % --- Back-propagation (unchanged) ---
        Or = MyAdjointOperatorPropagation(I, H);
        bp_ch = real(Or);

        % --- Propagation operators (unchanged) ---
        A  = @(obj) MyForwardOperatorPropagation(obj, H);
        AT = @(Iin) MyAdjointOperatorPropagation(Iin, H);

        % --- TwIST settings (unchanged) ---
        tv_iters = 2;
        Psi = @(x,th) tvdenoise(x, 2/th, tv_iters);
        Phi = @(x) TVnorm(x);
        tau = 0.005; tolA = 1e-6; iterations = 200;

        % --- TwIST reconstruction (unchanged) ---
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
        cs_ch = real(f_reconstruct);

        % --- (Plots commented out, identical to your script) ---
        % figure, imagesc(Im); title('Original image'); colormap gray; axis image off
        % figure, imagesc(mask); title('FZA pattern'); colormap gray; axis image off
        % figure, imagesc(I); title('Observed imaging'); colormap gray; axis image off
        % figure, imagesc(bp_ch); title('Reconstructed image (BP)'); colormap gray; axis image off
        % figure, imagesc(cs_ch); title('Reconstructed image (CS)'); colormap gray; axis image off
    end
end
