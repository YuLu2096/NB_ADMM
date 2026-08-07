%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%This function solves the following nonconvex variant with AITV regularization
%and nonnegative constraint via ADMM:
%
%   min <(R + g) log(R + v) - g log v, 1> +
%       tau (||w||_1 - alpha ||w||_{2,1}) + I_{>=0}(f)
%   s.t. Au = v, Du = w, f = u
%
%Input:
%   g: noisy image
%   A: blurring operator
%   alpha: sparsity parameter for L1-alpha L2 term of gradient
%   tau: regularization parameter tau in the paper
%   beta: initial ADMM penalty parameter beta^0 in the paper
%   r: negative binomial parameter
%
%Output:
%   f_sol: solution/recovered image (nonnegative)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [f_sol] = Deblur_NB_L1mL2(g, A, alpha, tau, beta, r)

    % penalty parameter multiplier (sigma in the paper)
    rho = 1.1;

    % max iterations (K in the paper)
    K = 300;

    % obtain dimension of image
    [rows, cols] = size(g);

    % preinitialize variables
    u0 = ones(rows, cols);
    u  = u0;
    f  = zeros(rows,cols);
    v  = u0;
    wx = u0;
    wy = u0;

    % preinitialize dual variables
    x  = ones(rows, cols);    % dual for Au = v
    yx = ones(rows, cols);    % dual for Dx(u) = wx
    yy = ones(rows, cols);    % dual for Dy(u) = wy
    z  = ones(rows, cols);    % dual for f = u

    % build kernel: use the fft algorithm (5-pt stencil)
    uker = zeros(rows, cols);
    uker(1,1) = 4; uker(1,2) = -1; uker(2,1) = -1; uker(rows,1) = -1; uker(1,cols) = -1;

    % refit blurring operator and shift it
    [xLen_flt, yLen_flt] = size(A);
    ope_blur = zeros(rows, cols);
    ope_blur(1:xLen_flt, 1:yLen_flt) = A;

    xLen_flt_1 = floor(xLen_flt/2); yLen_flt_1 = floor(yLen_flt/2);
    ope_blur_1 = padarray(ope_blur, [rows, cols], 'circular', 'pre');
    ope_blur_1 = ope_blur_1(xLen_flt_1+1:rows+xLen_flt_1, yLen_flt_1+1:cols+yLen_flt_1);

    % Fourier transform of blurring operator
    FA = fft2(ope_blur_1);

    % ADMM penalty parameter
    admm_beta = beta;  % beta^0 in the paper

    for k = 1:K

        % store past f for stopping criterion
        f_old = f;

        %----- f-subproblem (nonneg projection) -----
        % f^{k+1} = max(u^k - z^k / beta^k, 0)
        f = max(u - z / admm_beta, 0);

        %----- u-subproblem (FFT solve) -----
        % Normal equation: beta*(A'A + D'D + I) u = A'(beta*v - x) - D'(y - beta*w) + beta*f^{k+1} + z
        lhs_ker = admm_beta * conj(FA).*FA + admm_beta * fft2(uker) + admm_beta;

        rhs1 = conj(FA) .* fft2(admm_beta * v - x);
        rhs2 = admm_beta * Dxt(wx) - Dxt(yx) + admm_beta * Dyt(wy) - Dyt(yy);
        rhs3 = admm_beta * f + z;

        u = real(ifft2((rhs1 + fft2(rhs2) + fft2(rhs3)) ./ lhs_ker));

        % compute Au
        Au = real(ifft2(FA .* fft2(u)));

        %----- v-subproblem (cubic root) -----
        delta = admm_beta;
        c3 = delta;
        c2 = delta * r - delta * Au - x;
        c1 = -delta * Au * r - r * x + r;
        c0 = -g * r;

        a2 = c2 / c3;
        a1 = c1 / c3;
        a0 = c0 / c3;

        Q = (3 * a1 - a2 .^ 2) / 9;
        R_cubic = (9 * a2.*a1 - 27 * a0 - 2 * a2.^3) / 54;
        D = Q.^3 + R_cubic.^2;

        S_tem1 = R_cubic + sqrt(D);
        S = S_tem1.^(1/3);

        T_tem1 = R_cubic - sqrt(D);
        T = T_tem1.^(1/3);

        v1 = -1/3 * a2 + (S + T);
        v2 = -1/3 * a2 - (S + T)/2 + 1i/2 * sqrt(3) * (S - T);
        v3 = -1/3 * a2 - (S + T)/2 - 1i/2 * sqrt(3) * (S - T);

        if isreal(v2)
            v1(v1 < 0) = 10e-6;
            v2(v2 < 0) = 10e-6;
            v3(v3 < 0) = 10e-6;

            val1 = (r+g).*log(r+v1) - g.*log(v1) - x.*v1 + delta/2 * (Au-v1).^2;
            val2 = (r+g).*log(r+v2) - g.*log(v2) - x.*v2 + delta/2 * (Au-v2).^2;
            val3 = (r+g).*log(r+v3) - g.*log(v3) - x.*v3 + delta/2 * (Au-v3).^2;

            v1(val1 > val2) = 0;
            v1(val1 > val3) = 0;
            v2(val2 > val3) = 0;
            v2(val2 > val1) = 0;
            v3(val3 > val1) = 0;
            v3(val3 > val2) = 0;

            v = v1 + v2 + v3;
        else
            v = v1;
        end

        %----- w-subproblem (proximal operator) -----
        temp1 = Dx(u) + yx / admm_beta;
        temp2 = Dy(u) + yy / admm_beta;

        temp1 = reshape(temp1, rows*cols, 1);
        temp2 = reshape(temp2, rows*cols, 1);

        temp = [temp1, temp2];
        temp = shrinkL12(temp, tau/admm_beta, alpha);
        wx = reshape(temp(:,1), rows, cols);
        wy = reshape(temp(:,2), rows, cols);

        %----- dual variable updates -----
        x  = x  + admm_beta * (Au - v);           % dual for Au = v
        yx = yx + admm_beta * (Dx(u) - wx);       % dual for Dx(u) = wx
        yy = yy + admm_beta * (Dy(u) - wy);       % dual for Dy(u) = wy
        z  = z  + admm_beta * (f - u);             % dual for f = u

        %----- update ADMM penalty parameter -----
        admm_beta = admm_beta * rho;

        %----- stopping criterion -----
        err = norm(f - f_old, 'fro') / norm(f, 'fro');
        if err < 10^(-4)
            break;
        end
    end

    f_sol = f;

end

function x = shrinkL12(y, lambda, alpha)
    x = zeros(size(y));

    [max_y, idx_y] = max(abs(y'));
    max_y = max_y';
    idx_y = idx_y';
    new_idx_y = sub2ind(size(y), (1:size(y,1))', idx_y);

    case1_idx = max_y > lambda;
    case1_result = max(abs(y(case1_idx,:)) - lambda, 0) .* sign(y(case1_idx,:));
    norm_case1_result = sqrt(sum(case1_result.^2, 2));
    x(case1_idx,:) = ((norm_case1_result + alpha*lambda) ./ norm_case1_result) .* case1_result;

    case2_idx = logical((max_y <= lambda) .* (max_y >= (1-alpha)*lambda));
    x(new_idx_y(case2_idx)) = (max_y(case2_idx) + (alpha-1)*lambda) .* sign(y(new_idx_y(case2_idx)));
end
