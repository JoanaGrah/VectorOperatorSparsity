function [u, w] = primaldualOptimisation(f, K, p, alpha)
%% PRIMALDUALOPTIMISATION_LIVE Image Denoising with Vector Operator Sparsity Regularisation
% _Eva-Maria Brinkmann, Martin Burger, Joana Sarah Grah_
% 
% "Unified Models for Second-Order TV-Type Regularisation in Imaging - A 
% new Perspective Based on Vector Operators." _Journal of Mathematical Imaging 
% and Vision_. 2018.
% 
% This function performs primal-dual optimisation [1,2] for VOS image denoising 
% and calculates a solution of the following minimisation problem:
% 
% $\min_{u,w} \frac{1}{2} \Vert u - f \Vert_2^2 + \alpha R_{\beta}(u)$, $R_{\beta}(u) 
% = \Vert \nabla u - w \Vert_1 + \Vert b \Vert_1$, where $b = ( \sqrt{\beta_1} 
% \text{curl}(w), \sqrt{\beta_2} \text{div}(w), \sqrt{\beta_3} \text{sh}_1(w), 
% \sqrt{\beta_4} \text{sh}_2(w) )^T$
% 
% Adapting the notation in [1], we have: $x = (u,w)^T$, $G(x) = \frac{1}{2} 
% \Vert u - f \Vert_2^2$, $F(Kx) = \alpha R_{\beta}(u)$.
% 
% References
% 
% # Antonin Chambolle and Thomas Pock. "A first-order primal-dual algorithm 
% for convex problems with applications to imaging." _Journal of mathematical 
% imaging and vision_ 40.1 (2011): 120-145.
% # Tom Goldstein, Min Li, Xiaoming Yuan, Ernie Esser and Richard Baraniuk. 
% "Adaptive primal-dual hybrid gradient methods for saddle-point problems." _arXiv 
% preprint arXiv:1305.0546_ (2015).
%% Preallocation
[m,n] = size(f);
f = f(:);
u = zeros(m*n,1);
u_old = u;
w1 = zeros(m*n,1);
w1_old = w1;
w2 = zeros(m*n,1);
w2_old = w2;
x_old = [u_old; w1_old; w2_old];
y11 = zeros(m*n,1);
y11_old = y11;
y12 = zeros(m*n,1);
y12_old = y12;
y21 = zeros(m*n,1);
y21_old = y21;
y22 = zeros(m*n,1);
y22_old = y22;
y23 = zeros(m*n,1);
y23_old = y23;
y24 = zeros(m*n,1);
y24_old = y24;
y_old = [y11_old; y12_old; y21_old; y22_old; y23_old; y24_old];
Kx_old = zeros(6*m*n,1);
Kx_bar = zeros(6*m*n,1);
Kadjointy_old = zeros(3*m*n,1);
iter = 0;
stopCrit = 0;
%% Prox Operators
primalProx = @(x,tau) (x + tau * f) ./ (1 + tau);
dualProx = @(y,norm_y,alpha) y ./ max(1, norm_y/alpha);
%% Optimisation
while iter < p.maxIter && stopCrit == 0
    
    iter = iter + 1;
%% 
% Update (1): $y = (I + \sigma \partial F^*)^{-1} (y_{\text{old}} + \sigma 
% K\bar{x})$
    y11_tilde = y11_old + p.sigma * Kx_bar(1:m*n,1);
    y12_tilde = y12_old + p.sigma * Kx_bar(m*n+1:2*m*n,1);
    norm_y1_tilde = sqrt(y11_tilde.^2 + y12_tilde.^2);
    y11 = dualProx(y11_tilde,norm_y1_tilde,alpha);
    y12 = dualProx(y12_tilde,norm_y1_tilde,alpha);
    
    y21_tilde = y21_old + p.sigma * Kx_bar(2*m*n+1:3*m*n,1);
    y22_tilde = y22_old + p.sigma * Kx_bar(3*m*n+1:4*m*n,1);
    y23_tilde = y23_old + p.sigma * Kx_bar(4*m*n+1:5*m*n,1);
    y24_tilde = y24_old + p.sigma * Kx_bar(5*m*n+1:6*m*n,1);
    norm_y2_tilde = sqrt(y21_tilde.^2 + y22_tilde.^2 + y23_tilde.^2 + y24_tilde.^2);
    y21 = dualProx(y21_tilde,norm_y2_tilde,1);
    y22 = dualProx(y22_tilde,norm_y2_tilde,1);
    y23 = dualProx(y23_tilde,norm_y2_tilde,1);
    y24 = dualProx(y24_tilde,norm_y2_tilde,1);
    
    y = [y11; y12; y21; y22; y23; y24];
%% 
% Update (2): $x = (I + \tau \partial G)^{-1} (x_{\text{old}} - \tau K^*y)$
    Kadjointy = K' * y;
    
    u_tilde = u_old - p.tau * Kadjointy(1:m*n,1);
    u = primalProx(u_tilde,p.tau);
    
    w1 = w1_old - p.tau * Kadjointy(m*n+1:2*m*n,1);    
    w2 = w2_old - p.tau * Kadjointy(2*m*n+1:3*m*n,1);
    
    x = [u; w1; w2];
%% 
% Update (3): $\bar{x} = x + \theta (x - x_{\text{old}})$
% 
% Included in calculation of $K\bar{x}$
    Kx = K * x;
    Kx_bar = (1 + p.theta) * Kx - p.theta * Kx_old;
%% 
% Stopping Criteria
    if mod(iter,250) == 0
%% 
% (1): Primal-Dual Residual
% 
% Primal Residual: $\frac{x_{\text{old}} - x}{\tau} - (K^* y_{\text{old}} 
% - K^* y )$
        primalRes = (x_old - x)/p.tau - (Kadjointy_old - Kadjointy);    
%% 
% Dual Residual: $\frac{y_{\text{old}} - y}{\sigma} - (K x_{\text{old}} 
% - Kx)$
        dualRes = (y_old - y)/p.sigma - (Kx_old - Kx);   
%% 
% Primal-Dual Residual: $\Vert \text{primalRes} \Vert_1 + \Vert \text{dualRes} 
% \Vert_1$ (scaled)
        pdRes = norm(primalRes,1)/numel(x) + norm(dualRes,1)/numel(y);
%% 
% (2): Primal-Dual Gap
        primalProblem = 0.5 * norm(u - f)^2 + alpha * norm(sqrt(Kx(1:m*n,1).^2 + Kx(m*n+1:2*m*n,1).^2),1) + norm(sqrt(Kx_bar(2*m*n+1:3*m*n,1).^2 + Kx_bar(3*m*n+1:4*m*n,1).^2 + Kx_bar(4*m*n+1:5*m*n,1).^2 + Kx_bar(5*m*n+1:6*m*n,1).^2),1); 
        dualProblem = - (0.5 * norm(-Kadjointy(1:m*n,1))^2 + sum(f.*(-Kadjointy(1:m*n,1))));
        pdGap = abs((primalProblem - dualProblem) / primalProblem);
%% 
% Check Stopping Criteria
        disp(['Iteration ',num2str(iter),'. Primal-Dual Residual: ',num2str(pdRes),'. Primal-Dual Gap: ',num2str(pdGap)]);    
        if pdRes < p.epsilon && pdGap < p.epsilon
            stopCrit = 1;
        end 
    end
%% 
% Update variables
    u_old = u;
    w1_old = w1;
    w2_old = w2;
    x_old = [u_old; w1_old; w2_old];
    y11_old = y11;
    y12_old = y12;
    y21_old = y21;
    y22_old = y22;
    y23_old = y23;
    y24_old = y24;
    y_old = [y11_old; y12_old; y21_old; y22_old; y23_old; y24_old];
    Kx_old = Kx;
    Kadjointy_old = Kadjointy;
end
disp(['Total number of iterations: ',num2str(iter),'.']);
%% Return optimal u and w
u = reshape(u, [m,n]);
w = cat(3, reshape(w1, [m,n]), reshape(w2, [m,n]));
end
%%