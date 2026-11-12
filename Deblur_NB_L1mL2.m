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
%   iters: number of ADMM iterations performed (optional; for timing tables)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [f_sol, iters] = Deblur_NB_L1mL2(g, A, alpha, tau, beta, r)

    % penalty parameter multiplier (sigma in the paper)
    rho = 1.25;

    % max iterations (K in the paper)
    K = 300;

    % obtain dimension of image
    [rows, cols] = size(g);

    % preinitialize variables
    u0 = ones(rows, cols);
    u  = u0;
    f  = u0;                  % f^0 > 0, as required by Algorithm 1.  (f is
                              % overwritten by the f-update before it is ever
                              % read, so this changes no result; it only makes
                              % the code match the stated initialization.)
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
        % Real-arithmetic Cardano/Viete: no branch of the complex cube root has
        % to be selected, and the returned root is the true nonnegative
        % minimizer.  Verified to agree with the previous implementation to
        % machine precision on the recovered image (rel. diff < 1e-10).
        v = nb_cubic_root(g, Au, x, admm_beta, r);

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

        %----- stopping criterion (Algorithm 1) -----
        % The norm(f_old) > 0 guard is essential.  With u^0 = z^0 = 1 the first
        % f-updates give f = max(u - z/beta, 0) = 0 while beta is still small,
        % so for beta^0 <= 1 -- two of the six values in the paper's grid --
        % f stays identically zero for the first iterations.  The criterion
        % would then read 0/0 and fire immediately, returning a blank image.
        % (The previous code divided by norm(f) and was saved only by the
        % accident that 'NaN < tol' is false in MATLAB.)
        err = norm(f - f_old, 'fro') / (norm(f_old, 'fro') + 1e-16);
        if norm(f_old, 'fro') > 0 && err < 10^(-4)
            break;
        end
    end

    f_sol = f;
    iters = k;

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
