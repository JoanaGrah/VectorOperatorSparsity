%% Image Denoising with Vector Operator Sparsity Regularisation
% _Eva-Maria Brinkmann, Martin Burger, Joana Sarah Grah_
% 
% "Unified Models for Second-Order TV-Type Regularisation in Imaging - A 
% new Perspective Based on Vector Operators." _Journal of Mathematical Imaging 
% and Vision_. 2018.
% 
% This code performs denoising for a given image by using the unified VOS 
% model optimised by a primal-dual algorithm.
%% Load Trui test image and add Gaussian noise
%%
load Trui.mat
[m,n] = size(img);

noiseMean = 0;
noiseVariance = 0.05;
imgNoisy = imnoise(img,'gaussian',noiseMean,noiseVariance);
%% Set Parameters
%%
alpha = 1/4.5;                                 %global regularisation weighting parameter
beta1 = 0;                                     %weighting parameter curl
beta2 = 1/8;                                   %weighting parameter divergence
beta3 = 1;                                     %weighting parameter shear1
beta4 = 1/2;                                   %weighting parameter shear2

Dy = spdiags([-ones(m,1) ones(m,1)], [0 1], m, m);
Dy(m,:) = 0;
Dx = spdiags([-ones(n,1) ones(n,1)], [0 1], n, n);
Dx(n,:) = 0;
DX = kron(Dx, speye(m));                       %gradient matrices
DY = kron(speye(n), Dy);
I = speye(m*n);                                %operator K
Z = sparse(m*n,m*n);
K = [DX -I Z; DY Z -I; Z -beta1*DY beta1*DX; Z -beta2*DX' -beta2*DY'; Z beta3*DX' -beta3*DY'; Z beta4*DY beta4*DX];

algorithmParam.sigma = 1/sqrt(normest(K'*K));  %primal step size
algorithmParam.tau = algorithmParam.sigma;     %dual step size
algorithmParam.theta = 1;                      %primal-dual overrelaxation parameter
algorithmParam.maxIter = 15000;                %maximum number of primal-dual iterations
algorithmParam.epsilon = 1e-3;                 %primal-dual residual threshold
%% Optimisation
%%
[u, w] = primaldualOptimisation(imgNoisy, K, algorithmParam, alpha);

disp(['Relative Error: ',num2str(norm(img-u)/norm(img)),', PSNR: ',num2str(psnr(u,img)),', SSIM:',num2str(ssim(u,img))]);
%% Visualisation

w1 = w(:,:,1);
w2 = w(:,:,2);
norm_v = sqrt((reshape(DX*u(:),[m,n]) - w1).^2 + (reshape(DY*u(:),[m,n]) - w2).^2);
curl_w = reshape(DY'*w1(:) - DX'*w2(:),[m,n]);
div_w = reshape(-DX'*w1(:) -DY'*w2(:),[m,n]);
sh1_w = reshape(DY*w2(:) - DX*w1(:),[m,n]);
sh2_w = reshape(DY*w1(:) + DX*w2(:),[m,n]);

cmap = colormap('gray');
figure;
subplot(421);imagesc(img);title('Clean Image');axis off;axis image;colormap(gca,'gray');
subplot(422);imagesc(imgNoisy);title('Noisy Image');axis off;axis image;colormap(gca,'gray');
subplot(423);imagesc(u);title('Denoised Image');axis off;axis image;colormap(gca,'gray');
subplot(424);imagesc(norm_v);title('Norm of Sparse Vector Field v = \nabla u - w');axis off;axis image;colormap(gca,flipud(cmap));
subplot(425);imagesc(curl_w);title('curl(w)');colorbar;axis off;axis image;colormap(gca,'parula');
subplot(426);imagesc(div_w);title('div(w)');colorbar;axis off;axis image;colormap(gca,'parula');
subplot(427);imagesc(sh1_w);title('sh_1(w)');colorbar;axis off;axis image;colormap(gca,'parula');
subplot(428);imagesc(sh2_w);title('sh_2(w)');colorbar;axis off;axis image;colormap(gca,'parula');