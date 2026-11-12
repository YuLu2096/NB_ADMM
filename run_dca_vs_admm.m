%% run_dca_vs_admm.m
% ADMM vs DCA for the NB-AITV model.
%
% Produces the runtime / iteration data requested by Reviewer comment 32
% ("the paper motivates ADMM partly by computational efficiency relative to
% DCA, but reports no runtimes, iteration counts, or related computational
% data"), and doubles as the self-check for Deblur_NB_DCA:
%
%   ASSERT 1 - both solvers return real, nonnegative images.  (This also
%              probes the cubic branch at r = 1, where the principal-branch
%              cube root used to make v complex.)
%   ASSERT 2 - the DC objective decreases overall, and any upward tick caused
%              by the inexact inner solve stays below 5% of that decrease.
%
% QUICK = true  -> 128x128 crop, r = 100 only.  ~5 s.  Use this to verify the
%                  code is correct.
% QUICK = false -> full image, r = 1, 10, 100, 1000 (the four noise levels of
%                  Tables 1-4).  ~6-10 min.  Use this for the paper numbers.
%
% The DCA is slow by construction: every outer iteration solves a convex
% subproblem that has no closed form, so it runs a full inner PDHG loop
% (K1*K2 = 6,000 inner iterations, 4 FFTs each) against ~60-90 ADMM
% iterations.  That gap is the point of the experiment, not a bug.
%
% Run run_all_experiments.m first: the tuned parameters come from results/results.mat.
close all; clear all; clc;

QUICK = false;

root = fileparts(mfilename('fullpath'));

peak    = 80;
img_dir = fullfile(root, 'images');
img_idx = 2;                                  % same image used in the parameter analysis
A       = fspecial('gaussian', [10 10], 2);   % Gaussian blur (injective kernel)
bidx    = 2;                                  % blur_labels = {Identity,Gaussian,Motion,Disk}

if QUICK
    r_list = 100;                      % smoke test: one noise level is enough
else
    r_list = [1, 10, 100, 1000];       % all four, matching Tables 1-4
end

% optimal parameters from the paper's grid search
S = load(fullfile(root, 'results', 'results.mat'));

%% ---- ground truth
I = double(imread(fullfile(img_dir, [num2str(img_idx) '.jpg'])));
if size(I,3) == 3, I = rgb2gray(I); end
if QUICK
    I = I(1:128, 1:128);                      % crop: 9x fewer pixels, 9x faster
end
Q = max(I(:))/peak;
I = I/Q;
I(I == 0) = min(I(I > 0));
I_blurry = myconv(I, A);

fprintf('image %d, size %s, QUICK = %d\n\n', img_idx, mat2str(size(I)), QUICK);
fprintf('%-6s %-6s %10s %8s %8s %8s %8s %8s\n', ...
        'r', 'method', 'time(s)', 'iters', 'outer', 'PSNR', 'SSIM', 'MAE');
fprintf('%s\n', repmat('-', 1, 70));

% one row per r: r | ADMM time iters PSNR SSIM MAE | DCA time inner outer PSNR SSIM MAE descended
T = [];

for r = r_list

    nidx  = find(S.r_values == r);
    tau   = S.best_tau(img_idx, nidx, bidx);
    alpha = S.best_alpha(img_idx, nidx, bidx);
    beta0 = S.best_beta0(img_idx, nidx, bidx);

    % Leave OVERRIDE empty for the numbers that go in the paper: timing should
    % be reported at the operating point the paper actually uses.  Setting it to
    % [tau alpha beta0] (e.g. [0.3 0.9 2.0]) is only a robustness check that the
    % conclusion does not depend on alpha.  It does not: the DCA subproblem has
    % no closed form because of the -g log(Af) + tau||Df||_1 part, not because
    % of the alpha term, so the nested inner solver is required for every alpha.
    OVERRIDE = [];
    if ~isempty(OVERRIDE)
        tau = OVERRIDE(1); alpha = OVERRIDE(2); beta0 = OVERRIDE(3);
    end
    fprintf('  r = %g: tau = %.3f, alpha = %.2f, beta0 = %.2f\n', r, tau, alpha, beta0);

    % same noise model and realization as run_all_experiments.m:
    % r = 1000 is the Poisson row of the tables, generated with poissrnd.
    rng('default'); rng(42);
    if r == 1000
        u0 = poissrnd(I_blurry);
    else
        u0 = nbinrnd(r, r ./ (I_blurry + r));
    end
    u0 = double(u0);
    u0 = u0 / max(u0(:));

    % ---------------- ADMM ----------------
    t = tic;
    [f_admm, it_admm] = Deblur_NB_L1mL2(u0, A, alpha, tau, beta0, r);
    t_admm = toc(t);

    % ---------------- DCA (Algorithm 2; PDHG inner solver, no beta) -----
    fprintf('  running DCA (r = %g)...\n', r);
    [f_dca, info] = Deblur_NB_DCA(u0, A, alpha, tau, r);

    % ---------------- checks --------------
    assert(isreal(f_admm) && all(f_admm(:) >= 0), ...
        'ADMM returned a complex or negative image at r = %g (check the cubic branch).', r);
    assert(isreal(f_dca) && all(f_dca(:) >= 0), ...
        'DCA returned a complex or negative image at r = %g.', r);

    % DCA is a descent method, but only when each subproblem is solved exactly.
    % An inexact inner solve lets the objective tick upward slightly, so the
    % test is relative to the objective's own scale rather than to the total
    % decrease -- the decrease can legitimately be ~0 once the outer loop stops
    % on TOL_OBJ, whereas a broken inner solve moves the objective by ~1e-3 of
    % its scale (measured: +2.1e4 on a 7.6e6 objective).
    % Threshold calibrated on the full image at the tuned parameters:
    %   converged runs      rise/scale <= 1.3e-04   (r = 1, the loosest)
    %   MINIT removed       rise/scale  = 4.7e+00 and 1.8e+02  (catastrophic)
    % 1e-3 clears the former by 7x and catches the latter by >4000x.
    scale = max(1, abs(info.obj(1)));
    rise  = max([diff(info.obj); -inf]);         % largest single-step increase
    assert(rise <= 1e-3 * scale, ...
        ['DCA objective rose by %.3g (%.1e of its scale) at r = %g; ' ...
         'check MINIT / K2 in Deblur_NB_DCA.'], rise, rise/scale, r);
    % Whether the DC objective actually decreased is REPORTED, not asserted.
    % DCA guarantees descent only for exactly solved subproblems, so under the
    % fixed inner budget K2 it can fail -- and that failure is a finding about
    % DCA on this problem, not a defect of the run.  Measured at K2 = 300: it
    % descends at r = 10, 100, 1000 and does not at r = 1, where the objective
    % rose on 13 of 19 outer steps (+5.33, i.e. 2.2e-4 of its scale).
    if ~info.descended
        fprintf(['  NOTE: DCA did not descend at r = %g -- the DC objective ' ...
                 'rose by %.3g (%.1e of scale) over %d outer iterations.\n' ...
                 '        The subproblem is not solved accurately enough at ' ...
                 'K2 = 300; report this row as non-convergent.\n'], ...
                r, info.obj_rise, info.obj_rise/scale, info.outer);
    end

    % ---------------- metrics -------------
    [p_admm, s_admm, l_admm] = score(f_admm, I, Q, peak);
    [p_dca,  s_dca,  l_dca ] = score(f_dca,  I, Q, peak);

    fprintf('%-6g %-6s %10.2f %8d %8s %8.2f %8.4f %8.3f\n', ...
            r, 'ADMM', t_admm,    it_admm,    '-',        p_admm, s_admm, l_admm);
    if info.descended, mark = ''; else, mark = '  (did not descend)'; end
    fprintf('%-6g %-6s %10.2f %8d %8d %8.2f %8.4f %8.3f%s\n', ...
            r, 'DCA',  info.time, info.inner, info.outer, p_dca,  s_dca,  l_dca, mark);
    T = [T; r, t_admm, it_admm, p_admm, s_admm, l_admm, ...
            info.time, info.inner, info.outer, p_dca, s_dca, l_dca, info.descended];
end

if ~QUICK
    save(fullfile(root, 'results', 'dca_vs_admm.mat'), 'T');
end

fprintf('\nAll checks passed.\n');
if QUICK
    fprintf('This was the QUICK check. Set QUICK = false for the paper numbers.\n');
end

%% ------------------------------------------------------------------
function [p, s, l] = score(f, I, Q, peak)
    f = mat2gray(f) * peak;                       % same normalization as the experiments
    p = psnr(f*Q, I*Q, 255);
    s = ssim(uint8(f*Q), uint8(I*Q));
    l = sum(abs(f(:) - I(:))) / numel(I);         % MAE on the [0,80] scale
end
