%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Pixelwise solution of the v-subproblem
%
%   v = argmin_{v >= 0}  (r+g) log(r+v) - g log(v) - x v + (beta/2)(Au - v)^2
%
% whose stationarity condition is the cubic  P(v) = 0  with
%
%   v^3 + a2 v^2 + a1 v + a0 = 0,
%   a2 = r - Au - x/beta,  a1 = -r Au - r x/beta + r/beta,  a0 = -r g/beta.
%
% The roots are obtained in ENTIRELY REAL arithmetic, so no branch of the
% complex cube root has to be chosen:
%   * Delta > 0  : one real root, via real cube roots (nthroot).
%   * Delta <= 0 : three real roots, via Viete's trigonometric form.
% v = 0 is also admitted as a candidate, since it is a root of P when g = 0
% (convention 0 log 0 := 0).  Among the admissible candidates the one with
% the smallest objective value is returned.
%
% Input:
%   g    : observed image
%   Au   : A*u at the current iterate
%   x    : Lagrange multiplier for Au = v
%   beta : ADMM penalty parameter (scalar)
%   r    : negative binomial parameter
%
% Output:
%   v    : minimizer, same size as g, with v >= 0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function v = nb_cubic_root(g, Au, x, beta, r)

    a2 = r - Au - x/beta;
    a1 = -r*Au - r*x/beta + r/beta;
    a0 = -r*g/beta;

    % depressed cubic  t^3 + p t + q = 0  with  v = t - a2/3
    p = a1 - a2.^2/3;
    q = 2*a2.^3/27 - a2.*a1/3 + a0;
    Delta = (q/2).^2 + (p/3).^3;

    % --- Delta > 0: the single real root (real cube roots, never complex)
    sD = sqrt(max(Delta, 0));
    t1 = nthroot(-q/2 + sD, 3) + nthroot(-q/2 - sD, 3);

    % --- Delta <= 0: three real roots (Viete).  Delta <= 0 forces p <= 0.
    pn  = min(p, -eps);
    m   = 2*sqrt(-pn/3);
    arg = min(max((3*q)./(pn.*m), -1), 1);   % clamp against round-off
    th  = acos(arg)/3;

    three = (Delta <= 0);
    shift = a2/3;

    cand = cat(3, ...
        (~three).*t1 + three.*(m.*cos(th))            - shift, ...
        (~three).*t1 + three.*(m.*cos(th - 2*pi/3))   - shift, ...
        (~three).*t1 + three.*(m.*cos(th + 2*pi/3))   - shift, ...
        zeros(size(g)));

    % --- keep the admissible candidate with the smallest objective
    best = inf(size(g));
    v    = zeros(size(g));
    for k = 1:size(cand, 3)
        vk = cand(:,:,k);
        hk = nb_obj(vk, g, Au, x, beta, r);
        take = hk < best;
        best(take) = hk(take);
        v(take)    = vk(take);
    end
end

% objective of the v-subproblem; +Inf outside the feasible set
function h = nb_obj(v, g, Au, x, beta, r)
    h  = inf(size(v));
    ok = (v > 0);
    h(ok) = (r + g(ok)).*log(r + v(ok)) - g(ok).*log(v(ok)) ...
            - x(ok).*v(ok) + beta/2*(Au(ok) - v(ok)).^2;

    % v = 0 is finite only when g = 0, using 0 log 0 := 0
    z = (v == 0) & (g == 0);
    h(z) = r*log(r) + beta/2*Au(z).^2;
end
