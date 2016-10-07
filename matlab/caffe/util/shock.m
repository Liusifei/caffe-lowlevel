function I=shock(I0,iter,dt,h,meth,Par)
%% evolve image according to a "shock filter" process
%% private function (by Guy Gilboa)
%% input: image,  #iterations, dt, h, method, parameters
%% output: evolved image
%% dt - time step size (default = 0.1)

%% h  - size of grid steps (default = 1)
%% Method: 'org' - original (Osher-Rudin)
%%         'alv' - Alavarez-Mazorra  
%%         'cmp' - Complex valued (Gilboa-Sochen-Zeevi)
%% Parameters:
%% 'alv': [c,sigma2]
%%        c - amount of diffusion in level-set direction
%%        sigma2 - variance of Gaussian smoothing in gradient direction
%% 'cmp': [lam,lam_tld,a,theta]
%%        lam - magnitude of complex diffusion in gradient direction
%%        lam_tld - amount of diffusion in level-set direction
%%        a -   slope of arctan (soft sign)
%%        theta - phase angle of complex part (default pi/1000)
%% example: I=shock(I0,iter,dt,h,'cmp',[lam,lam_tld,a])
%%


if ~exist('dt','var')
   dt=0.1;
end
if ~exist('h','var')
   h=1;
end
if ~exist('meth','var')
   meth='org';
end
if ~exist('iter','var')
    iter = 30;
end
if ((meth =='cmp') & (length(Par)<4))
   Par(4)=pi/1000;  % default theta value
end

[ny,nx]=size(I0);
I=I0;
h_2 = h^2;
% compute flow
for i=1:iter,  %% do iterations
   % estimate derivatives (Newmann BC)
   I_mx = I-I(:,[1 1:nx-1]);
   I_px = I(:,[2:nx nx])-I;
   I_my = I-I([1 1:ny-1],:);
   I_py = I([2:ny ny],:)-I;
   I_x = (I_mx+I_px)/2;
   I_y = (I_my+I_py)/2;
   % minmod operator
   Dx = min(abs(I_mx),abs(I_px));
   ind=find(I_mx.*I_px < 0); Dx(ind)=zeros(size(ind));
   Dy = min(abs(I_my),abs(I_py));
   ind=find(I_my.*I_py < 0); Dy(ind)=zeros(size(ind));
   
	I_xx = I(:,[2:nx nx])+I(:,[1 1:nx-1])-2*I;
   I_yy = I([2:ny ny],:)+I([1 1:ny-1],:)-2*I;
   I_xy = (I_x([2:ny ny],:)-I_x([1 1:ny-1],:))/2;
   
   % compute flow
   a_grad_I = sqrt(Dx.^2+Dy.^2); % Abs Gradient of I
     
   dl=0.00000001;  % small delta
   I_nn = I_xx.*abs(I_x).^2 + 2*I_xy.*I_x.*I_y + I_yy.*abs(I_y).^2;
   I_nn = I_nn./(abs(I_x).^2+abs(I_y).^2+dl);
   I_ee = I_xx.*abs(I_y).^2 - 2*I_xy.*I_x.*I_y + I_yy.*abs(I_x).^2;
   I_ee = I_ee./(abs(I_x).^2+abs(I_y).^2+dl);
   a2_grad_I = (abs(I_x)+abs(I_y)); % second order abs grad
   ind = find(a2_grad_I==0);  % zero gradient n,e not defined
   I_nn(ind)=I_xx(ind); 
   I_ee(ind)=I_yy(ind);

   if (meth=='org')
  		I_t = -sign(I_nn).*a_grad_I/h;
   elseif (meth=='cmp')
      lam=Par(1); lam_tld=Par(2); a=Par(3); theta=Par(4);
      i=sqrt(-1); lam=lam*exp(i*theta);  
      I_t = -atan(a/theta*imag(I)).*a_grad_I/h + lam*I_nn/h_2 + lam_tld*I_ee/h_2;
   elseif (meth=='alv')
   	c=Par(1); sigma2=Par(2);
     
      g_I_nn=gauss(I_nn,51,sigma2);  
      I_t = - sign(g_I_nn).*a_grad_I/h + c*I_ee/h_2;
   else 
      error(['unknown method ' meth])
   end  % if
      
   I=I+dt*I_t;  %% evolve image by dt   
   
end % for i