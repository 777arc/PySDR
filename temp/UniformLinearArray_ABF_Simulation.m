% Adaptive Beamforming Uniform Linear Array simulation
% Set up parameters
clear all, close all
nfft = 2048; % nfft for plotting section
N=30; % Number of adaptive degrees of freedom (DOFs) or input channels where N > 1
Kmax=floor(2*N);% Set number of realizations to create of training-data vectors aka "snapshots" 

%%%%%%%%%%%%%%% Build jammer covariance matrix %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%		J= number of jammers
%		SJ2DB= J vector of jammer powers in db
%		PHIDJ= J vector of jammer angles in degrees
%		G= N vector gains in db
%		fb= the fractional bandwidth = bandwidth/center freq.

look_direction = 0; % Set array beam position: 0 = broadside (centered) or else +-(integer less than N/2) moves beam around
s=exp(1i*(2*pi/N)*look_direction*[0:N-1]).'; % 
s=s/norm(s); % normalize steering vector - makes the expression s's = 1 mean that s has unit power per element (not unit energy in total) Thus, a*s has power per array element of p = |a*s|^2 = |a|^2 *1 = |a|^2

% Set # jammers and jammer parameters
J=3; SJ2DB=[30 30 30]; THETADJ= [-70 -20 40]; PHIDJ=(pi*sin(THETADJ*(pi/180)))*(180/pi); G=zeros(1,N); fb=0; % List jammer powers and angles in order from neg. to pos.
%J=5; SJ2DB=[30 10 30 50 30]; THETADJ= [-30 -20 -10 5 50]; PHIDJ=(pi*sin(THETADJ*(pi/180)))*(180/pi); G=zeros(1,N); fb=0; % List jammer powers and angles in order from neg. to pos. 

% Create jammer covariance matrix
Rx_array = Covmatrix2(N,J,SJ2DB,PHIDJ,G,fb);
%  Rx_array=eye(N); % uncomment this for zero jammers

% Calculate sqrt of covariance matrix used to 'color' random vectors to match jammer covariance matrix
A=Rx_array^(.5); % Effective Cholesky factoriz. of correlation matrix: produces the Hermitian of the result textbooks might produce
INV_Rx_array = inv(Rx_array); % Inverse of the ideal covariance matrix
sinr_opt = real(s'*INV_Rx_array*s); % 'real' used to clean up (remove) neglible imaginary component caused by finite precision Covariance matrix inversion 

% Create correlated data matrix c1 which is generated from the jammer covariance matrix
c1= NonStatData(A,N,Kmax+1,0); % dimensions are N x Kmax. There are Kmax independent vector realizations of a random vector having jammer covariance matrix
R_smi = (c1*c1')/Kmax; % Calculate Sample Covariance Matrix
INV_R_smi = inv(R_smi); % Inverse of Sample Covariance Matrix
SINR_adaptive = real(s'*(INV_R_smi)*s); % 'real' used to clean up (remove) neglible imaginary component caused by finite precision Covariance matrix inversion 

% Plot Eigenspectra for the data realization identified by IF statement immediately below
Fsize = 14; % specify font size for plots
[a b]=sort(10*log10(real(eig((1/size(c1,2))*(c1*c1'))))); % Sampled jammer covariance matrix (note: sorting BEFORE 10*log10 function results in inaccurate sort)
for i=1:N, aa1(N+1-i)=a(i); , end, figure, plot(aa1, 'k*')
xlabel('Eigenvalue Number', 'FontWeight', 'bold', 'FontSize', Fsize), ylabel('Eigenvalue, dB', 'FontWeight', 'bold', 'FontSize', Fsize), axis([0 N (min(aa1)-10) (max(aa1) +10)]), grid
title(['Eigenspectra of true & sampled jammer covariance matrix, using K = ', num2str(Kmax), ' samples']);
hold on
[a b]=sort(10*log10(abs(real(eig(Rx_array))))); % True jammer covariance matrix (note: sorting BEFORE 10*log10 function results in inaccurate sort)
for i=1:N, aa2(N+1-i)=a(i); , end, plot(aa2, 'r')
legend('Sampled','True')

% Non-adaptive solution
w_non_adaptive=s; % used '/sqrt(N)' to force peak to zero since fft kernel is not normalized to unit norm (see below)
NON_ADAPTIVE=20*log10(fftshift(abs(fft(w_non_adaptive,nfft))/sqrt(N))); % used '/sqrt(N)' to force peak to zero since fft kernel is not normalized to unit norm (see below)

% Optimal or Ideal solution (just stick with the term optimal, be consistent)
w_ideal = (INV_Rx_array*s)/(s'*INV_Rx_array*s);
IDEAL = 20*log10(fftshift(abs(fft(w_ideal,nfft))/sqrt(N))); % Divide by sqrt(N) since fft kernel is not normalized to unit norm like s is so dividing by sqrt(N) effectively normalized the fft kernel so the max output is 1 or 0 dB

% Adaptive solution
w_adaptive = (INV_R_smi*s)/(s'*INV_R_smi*s); % MVDR ABF solution
ADAPTIVE = 20*log10(fftshift(abs(fft(w_adaptive,nfft))/sqrt(N))); % Divide by sqrt(N) since fft kernel is not normalized to unit norm like s is so dividing by sqrt(N) effectively normalized the fft kernel so the max output is 1 or 0 dB

% The adaptive solution's pattern will "calm down", as you add more samples. it will always have unity gain, but the sidelobes will come down
% for N=10 try doing a 20 samples vs 100 samples and looking at the sidelobe levels
% The quienscent pattern *is* the conventional beamformer (aka nonadaptive beamforming) when your weights = s, fix it earlier in the chapter
% when you take the fft to find the pattern, just call that calculating the beam pattern
% eg "adaptive beam pattern", "non-adaptive beam pattern"

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Plots  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Plot optimal and adaptive antenna patterns
faxis=get_fft_axis(nfft,2*pi,1); % since unambig range = -pi:pi (i.e., 2pi) sample rate is twice this or 4pi ?? rad/sample
faxis_physical_angle = asin(faxis/pi) * (180/pi);
h0=figure; set(h0,'DefaultLineLineWidth',2);

plot(faxis_physical_angle,NON_ADAPTIVE, 'b'), hold on
plot(faxis_physical_angle,IDEAL, 'r')
plot(faxis_physical_angle,ADAPTIVE, 'k')

ax1 = [-90 90 -80 10];
xlabel('Azimuth angle (deg)', 'FontWeight', 'bold', 'FontSize', Fsize), ylabel('Power, dB', 'FontWeight', 'bold', 'FontSize', Fsize)
title(['N=',num2str(N),' DOF, K=', num2str(Kmax),', SINR_o_p_t= ',num2str(10*log10(sinr_opt)),' dB, SINR_a_d_a_p_t= ', num2str(10*log10(SINR_adaptive)), ' dB'], 'FontWeight', 'bold', 'FontSize', Fsize)
axis(ax1);
set(h0,'DefaultLineLineWidth',1);
for ii=1: length(THETADJ)
    plot([THETADJ(ii) THETADJ(ii)], [ax1(3) ax1(4)],'g')
end
legend('Non-Adaptive Pattern','Optimal Pattern','Adaptive Pattern', ...
    ['Jammer locations: JNR = ', num2str(SJ2DB),' dB  (L to R),  Fractional BW = ', num2str(fb)], ...
    'Location', 'SouthWest')
legend boxoff
set(gca, 'FontWeight', 'bold', 'FontSize', Fsize)
grid
      