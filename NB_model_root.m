function [v1, v2, v3] = NB_model_root(c3, c2, c1, c0)
    a2 = c2 / c3;
    a1 = c1 / c3;
    a0 = c0 / c3;

    Q = (3 * a1 - a2 .^ 2) / 9;

    R = (9 * a2.*a1 - 27* a0 - 2* a2.^ 3) / 54;

    D = Q.^ 3 + R.^ 2;

    tem1 = R + sqrt(D);
    if isreal(tem1)
        S = nthroot(R + sqrt(D),3);
    else
        S = (R + sqrt(D)).^(1/3);
    end

    tem2 = R - sqrt(D);
    if isreal(tem2)
        T = nthroot(R - sqrt(D),3);
    else
        T = (R - sqrt(D)).^(1/3);
    end



    v1 = -1/3 * a2 + (S + T);
    v2 = -1/3 * a2 - (S + T)/2 + 1i/2 * sqrt(3) *(S - T);
    v3 = -1/3 * a2 - (S + T)/2 - 1i/2 * sqrt(3) *(S - T);
end