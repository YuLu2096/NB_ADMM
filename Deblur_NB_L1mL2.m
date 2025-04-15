%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%This function solves the following nonconvex variant of the Mumford-Shah
%model with AITV regularization:
%
%   min \langle (R + f) log(R + Au) - f log Au \rangle +
%   beta \|\nabla u\|_1 - \alpha \|\nabla u\|_{2,1}
%
%Input:
%   f: noisy image
%   A: deblurring operator
%   beta: weighing parameter for fidelity term
%   alpha: sparsity parameter for L1-\alpha L2 term of gradient
%   tau: penalty parameter for ADMM
%   mu: weighing parameter for smoothing term (% In this paper, mu = 0.)
%
%Output:
%   u: solution/smoothed image
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [u]= Deblur_NB_L1mL2(f, A, mu, alpha, beta, tau, r)
    
    %penalty parameter multiplier
    rho = 1.1;
    %tau = 0.1;
    
    %obtain dimension of image
    [rows,cols] = size(f);
    
    %preintialize variable to store past u
    u0 = ones(rows,cols);
    
    %preinitialize u
    u= u0;

    %preinitialize v
    v=u0;
    
    %preinitialize w variables
    wx = u0;
    wy = u0;
    
    %preinitialize dual variable
    z = v;
    zx = u0;
    zy = u0;
    
    %build kernel: use the fft algorithm (5-pt stencil)
    uker = zeros(rows,cols);
    uker(1,1) = 4;uker(1,2)=-1;uker(2,1)=-1;uker(rows,1)=-1;uker(1,cols)=-1;
    
    %refit blurring operator and shift it
    [xLen_flt, yLen_flt] = size(A);
    ope_blur=zeros(rows,cols);
    ope_blur(1:xLen_flt,1:yLen_flt)=A;
    
    xLen_flt_1=floor(xLen_flt/2);yLen_flt_1=floor(yLen_flt/2);
    ope_blur_1=padarray(ope_blur,[rows,cols],'circular','pre');
    ope_blur_1=ope_blur_1(xLen_flt_1+1:rows+xLen_flt_1,yLen_flt_1+1:cols+yLen_flt_1);
    
    %fourier transform of blurring operator
    FA = fft2(ope_blur_1);

    %compute Au
    Au = ifft2(FA.*fft2(u));
    
    for i=1:300

 %store past u
        u0 = u;
        
        %left-hand side of optimality eqn of u
        new_uker = beta*conj(FA).*FA+(mu+beta)*fft2(uker);
        
        %right-hand side of optimality eqn of u
        rhs1 = beta*v-z;
        rhs2 = beta*Dxt(wx)-Dxt(zx)+beta*Dyt(wy)-Dyt(zy);
        
        %solve u-subproblem
        u = ifft2((conj(FA).*fft2(rhs1)+fft2(rhs2))./new_uker);

        %compute Au
        Au = ifft2(FA.*fft2(u));
        
        %compute relative err
        err=norm(u-u0,'fro')/norm(u, 'fro');
        
        % if mod(i,10)==0
        %     disp(['iterations: ' num2str(i) '!  ' 'error is:   ' num2str(err)]);
        % end
        
        % check the stopping criterion
        if err<10^(-4)
            break;
        end


        %solve v-subproblem beta=delta;x=z;Au=u
        delta=beta;
        c3 = delta; %coefficient of order 3
        c2 = delta * r - delta * Au - z; %coefficient of order 2, r is the NB parameter,
        c1 = -delta * Au * r - r * z +  (1/tau)*r; %coefficient of order 1
        c0 = -f * r * (1/tau);
        

        a2 = c2 / c3;
        a1 = c1 / c3;
        a0 = c0 / c3;

        Q = (3 * a1 - a2 .^ 2) / 9;
        R = (9 * a2.*a1 - 27* a0 - 2* a2.^ 3) / 54;
        D = Q.^ 3 + R.^ 2;


        S_tem1 = R + sqrt(D);
        S_tem2 = S_tem1;
        S_tem2(imag(S_tem2)==0) = 0; %set all real component = 0
        S_tem1(imag(S_tem1)~=0) = 0; %set all complex componet = 0
        S = nthroot(S_tem1,3) + S_tem2.^(1/3);


        T_tem1 = R - sqrt(D);
        T_tem2 = T_tem1;
        T_tem2(imag(T_tem2)==0) = 0; %set all real component = 0
        T_tem1(imag(T_tem1)~=0) = 0; %set all complex componet = 0
        T = nthroot(T_tem1,3) + T_tem2.^(1/3);

        % %three solutions
        v1 = -1/3 * a2 + (S + T); %always postive real
        v2 = -1/3 * a2 - (S + T)/2 + 1i/2 * sqrt(3) *(S - T);
        v3 = -1/3 * a2 - (S + T)/2 - 1i/2 * sqrt(3) *(S - T);


        if isreal(v2)
            v1(v1<0) = 10e-6;
            v2(v2<0) = 10e-6;
            v3(v3<0) = 10e-6;

            val1 = sum((r+f).*log(r+v1) - f.*log(v1)) + z .* (Au-v1) + delta/2 * (Au-v1).^2;
            val2 = sum((r+f).*log(r+v2) - f.*log(v2)) + z .* (Au-v2) + delta/2 * (Au-v2).^2;
            val3 = sum((r+f).*log(r+v3) - f.*log(v3)) + z .* (Au-v3) + delta/2 * (Au-v3).^2;

            v1(val1>val2) = 0;
            v1(val1>val3) = 0;

            v2(val2>val3) = 0;
            v2(val2>val1) = 0;

            v3(val3>val1) = 0;
            v3(val3>val2) = 0;

            v = v1+v2+v3;
        else
            v = v1;
        end 

        

        %solve w-subproblem
        temp1 = Dx(u)+zx/beta;
        temp2 = Dy(u)+zy/beta;
        
        temp1 = reshape(temp1, rows*cols,1);
        temp2 = reshape(temp2, rows*cols,1);
        
        temp = [temp1, temp2];
        temp = shrinkL12(temp,1/beta, alpha);
        wx = temp(:,1);
        wy = temp(:,2);
        wx = reshape(wx, rows,cols);
        wy = reshape(wy, rows,cols);
        
        %update dual variables
        zx = zx+beta*(Dx(u)-wx);
        zy = zy+beta*(Dy(u)-wy);
        z = z+beta*(Au-v);
        
        %update ADMM penalty parameter
        beta = beta*rho;
    end


end

function x = shrinkL12(y,lambda,alpha)
    %this function applies the proximal operator of L1-alpha L2 to each
    %row vector
    
    %initialize solution as zero vector
    x = zeros(size(y));
    
    %obtain the indices of the max entries of each row vector
    [max_y, idx_y] = max(abs(y'));
    max_y = max_y';
    idx_y = idx_y';
    new_idx_y = sub2ind(size(y), (1:size(y,1))',idx_y);
    
    %compute new row vectors when max value of each row vector is greater
    %than lambda
    case1_idx = max_y > lambda;
    
    case1_result = max(abs(y(case1_idx,:))-lambda,0).*sign(y(case1_idx,:));
    norm_case1_result = sqrt(sum(case1_result.^2,2));
    x(case1_idx,:) =((norm_case1_result+alpha*lambda)./norm_case1_result).*case1_result;
    
    %compute one-sparse vector when max value of each row vector is less
    %than or equal to lambda and above (1-alpha)*lambda
    case2_idx = logical((max_y<=lambda).*(max_y>=(1-alpha)*lambda));
    
    x(new_idx_y(case2_idx)) = (max_y(case2_idx)+(alpha-1)*lambda).*sign(y(new_idx_y(case2_idx)));
    
end


    
