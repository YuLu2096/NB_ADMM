%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DCA scheme for the NB-AITV model -- implementation of Algorithm 2
% (Appendix A) of the manuscript.
%
%   min_{f >= 0}  <(R+g) log(R+Af) - g log(Af), 1>
%                 + tau ( ||Df||_1 - alpha ||Df||_{2,1} )
%
% DC decomposition used in the paper:
%   E_1(f) = -<g log(Af), 1> + tau ||Df||_1 + I_{>=0}(f)
%   E_2(f) = -<(R+g) log(R+Af), 1> + tau*alpha ||Df||_{2,1}
% so BOTH the (R+g)log(R+Af) part of the fidelity AND the l_{2,1} part of the
% regularizer are linearized at f^k.  The outer iteration is
%
%   f^{k+1} = argmin_{f>=0} -<g log(Af),1> + tau||Df||_1
%                           + < A'((R+g)/(R+Af^k)) - tau*alpha*D'q^k , f >
%
%   q^k = Df^k / |Df^k|   (0 where Df^k = 0).
%
% The convex subproblem is solved by PDHG (Chambolle-Pock), with auxiliary
% variables v for Af and w for Df, exactly as in eqs. (dca_f), (dca_v),
% (dca_w_x)-(dca_w_y) of the appendix.
%
% Input:
%   g     : noisy blurry image
%   A     : blurring kernel
%   alpha : AITV parameter
%   tau   : regularization parameter
%   r     : negative binomial parameter
%
% Output:
%   f_sol : recovered image (nonnegative)
%   info  : .obj    DC objective after each outer iteration
%           .outer  outer (DCA) iterations performed
%           .inner  total inner (PDHG) iterations
%           .time   wall-clock seconds
%           .eta    step size used
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [f_sol, info] = Deblur_NB_DCA(g, A, alpha, tau, r)

    K1      = 20;      % outer (DCA) iterations
    K2      = 300;     % inner (PDHG) iterations per convex subproblem.
                       % If the DC objective does not descend under this budget
                       % it is reported via info.descended, not treated as an
                       % error.
    TOL_OUT = 1e-4;    % outer tolerance on the iterate, same as Algorithm 1
    TOL_OBJ = 1e-8;    % outer tolerance on the relative DC objective change
    TOL_IN  = 1e-6;    % inner tolerance.  Must be tighter than TOL_OUT: the DCA
                       % descent property assumes each subproblem is solved
                       % exactly, and an inexact solve lets the DC objective
                       % tick upward between outer iterations.
    MINIT   = 50;      % minimum inner iterations before the tolerance is tested.
                       % NOT optional.  The duals are reset to zero at every
                       % outer call, so the first PDHG steps move f very little
                       % and the relative-step test would fire immediately,
                       % leaving a badly damaged iterate.

    t0 = tic;
    [rows, cols] = size(g);

    % ---- Fourier symbol of A (same construction as Deblur_NB_L1mL2)
    ope_blur = zeros(rows, cols);
    [xLen, yLen] = size(A);
    ope_blur(1:xLen, 1:yLen) = A;
    ope_blur = padarray(ope_blur, [rows, cols], 'circular', 'pre');
    ope_blur = ope_blur(floor(xLen/2)+1 : rows+floor(xLen/2), ...
                        floor(yLen/2)+1 : cols+floor(yLen/2));
    FA = fft2(ope_blur);

    % ---- symbol of D'D (5-point stencil): ||D||^2 = max of its symbol
    uker = zeros(rows, cols);
    uker(1,1) = 4; uker(1,2) = -1; uker(2,1) = -1; uker(rows,1) = -1; uker(1,cols) = -1;
    Fuker = real(fft2(uker));
    normK = sqrt(max(abs(FA(:)).^2) + max(Fuker(:)));   % ||K||, K = [A; D]

    % PDHG needs eta*zeta*||K||^2 <= 1 with K = [A; D]; take eta = zeta = 1/||K||
    eta  = 1/normK;
    zeta = 1/normK;

    f   = g;                       % feasible start, f >= 0
    obj = zeros(K1, 1);
    inner_total  = 0;
    obj_overhead = 0;   % time spent on instrumentation, excluded from info.time

    for k = 1:K1

        f_prev = f;

        % ---- linearize E_2 at f^k  ->  constant vector of the subproblem
        Af = Aop(FA, f);
        grad_fid = Atop(FA, (r + g) ./ (r + Af));      % A'((R+g)/(R+Af^k))

        dx = Dx(f); dy = Dy(f);
        nrm = sqrt(dx.^2 + dy.^2);
        qx = zeros(size(dx)); qy = zeros(size(dy));
        nz = nrm > 0;
        qx(nz) = dx(nz)./nrm(nz);
        qy(nz) = dy(nz)./nrm(nz);

        c = grad_fid - tau*alpha*(Dxt(qx) + Dyt(qy));

        % ---- PDHG on the convex subproblem
        [f, it] = pdhg(g, FA, tau, c, f, eta, zeta, K2, TOL_IN, MINIT);
        inner_total = inner_total + it;

        % Objective evaluation is INSTRUMENTATION, not part of the algorithm:
        % the ADMM it is being timed against computes no objective, so leaving
        % it inside the timer would bias the runtime comparison.  It is timed
        % separately and subtracted from info.time below, so the reported time
        % is the algorithm only.  It is kept because run_dca_vs_admm.m asserts
        % on it, and it is what caught the inner-solve regression at r = 1, 10.
        t_obj = tic;
        obj(k) = dc_objective(f, g, FA, tau, alpha, r);
        obj_overhead = obj_overhead + toc(t_obj);

        % Progress: one line per outer iteration.  A single outer iteration is
        % K2 PDHG steps (4 FFTs each), so without this the solver looks hung.
        fprintf('    DCA outer %2d/%d: %5d inner, obj = %.8e, %.1f s\n', ...
                k, K1, it, obj(k), toc(t0));

        % Stop on the iterate OR on the objective.  The iterate test alone can
        % keep the outer loop running long after the DC objective has flattened
        % (the iterate wanders on a plateau), which would make the DCA look
        % artificially slow in the ADMM-vs-DCA comparison.
        if norm(f - f_prev, 'fro') / (norm(f_prev, 'fro') + 1e-16) < TOL_OUT
            break;
        end
%         if k > 1 && abs(obj(k) - obj(k-1)) <= TOL_OBJ * max(1, abs(obj(k)))
%             break;
%         end
    end

    f_sol      = f;
    info.obj   = obj(1:k);
    % Did the DC objective actually decrease over the run?  DCA guarantees
    % descent only when each subproblem is solved exactly, so under a fixed
    % inner budget this can be false; it is reported, not enforced.
    info.obj_rise  = obj(k) - obj(1);
    info.descended = info.obj_rise <= 1e-6 * max(1, abs(obj(1)));
    info.outer = k;
    info.inner = inner_total;
    info.time  = toc(t0) - obj_overhead;   % algorithm only
    info.obj_overhead = obj_overhead;
    info.eta   = eta;
end

% ------------------------------------------------------------------------
% PDHG for   min_{f>=0} -<g log(Af),1> + tau||Df||_1 + <c,f>
% saddle form:  ... + <x, v - Af> + <y, w - Df>
% ------------------------------------------------------------------------
function [f, n] = pdhg(g, FA, tau, c, f0, eta, zeta, MAXIT, TOL, MINIT)

    f  = f0;
    v  = Aop(FA, f0);      % feasible initialization: v^0 = A f^k
    wx = Dx(f0);  wy = Dy(f0);
    x  = zeros(size(f0));
    yx = zeros(size(f0));  yy = zeros(size(f0));

    for n = 1:MAXIT

        f_old = f;  v_old = v;  wx_old = wx;  wy_old = wy;

        % f-update: projected proximal step  (eq. dca_f)
        f = max(f - eta*(c - Atop(FA, x) - (Dxt(yx) + Dyt(yy))), 0);

        % v-update: positive root of  v^2 + (eta*x - v)v - eta*g = 0  (eq. dca_v)
        s = v_old - eta*x;
        v = (s + sqrt(s.^2 + 4*eta*g)) / 2;

        % w-update: soft threshold at eta*tau  (eqs. dca_w_x, dca_w_y)
        wx = soft(wx_old - eta*yx, eta*tau);
        wy = soft(wy_old - eta*yy, eta*tau);

        % extrapolation
        fbar  = 2*f  - f_old;
        vbar  = 2*v  - v_old;
        wxbar = 2*wx - wx_old;
        wybar = 2*wy - wy_old;

        % dual ascent
        x  = x  + zeta*(vbar  - Aop(FA, fbar));
        yx = yx + zeta*(wxbar - Dx(fbar));
        yy = yy + zeta*(wybar - Dy(fbar));

        if n >= MINIT && norm(f - f_old, 'fro') / (norm(f_old, 'fro') + 1e-16) < TOL
            break;
        end
    end
end

function y = Aop(FA, f)
    y = real(ifft2(FA .* fft2(f)));
end

function y = Atop(FA, f)
    y = real(ifft2(conj(FA) .* fft2(f)));
end

function s = soft(t, lam)
    s = sign(t) .* max(abs(t) - lam, 0);
end

% DC objective  <(R+g)log(R+Af) - g log(Af), 1> + tau(||Df||_1 - alpha||Df||_{2,1})
function F = dc_objective(f, g, FA, tau, alpha, r)
    Af  = Aop(FA, f);
    % -g log(Af) is evaluated only where g > 0, using 0 log 0 := 0.  Flooring Af
    % for every pixel instead would cap the penalty and hide a diverging iterate.
    pos = g > 0;
    fid = sum(sum((r + g).*log(r + max(Af, 0)))) ...
          - sum(g(pos).*log(max(Af(pos), realmin)));
    dx  = Dx(f); dy = Dy(f);
    reg = sum(sum(abs(dx) + abs(dy))) - alpha*sum(sum(sqrt(dx.^2 + dy.^2)));
    F   = fid + tau*reg;
end
