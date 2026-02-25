% Adaptive spatial array simulation: Used to create training data to test performance of adaptive algorithms

% When I switch
% Barrage noise is wideband in the sense that it spans your bw of the rx
% By using a high fractional bandwth, you can simulate a jammer that is wider bw than what you are simulating
% which ends up making the jammer look like multiple jammers near (but not on top)

% On the last plot, if its not too busy, i can add MVDR using the perfect R 

% Set up parameters
clear
nfft = 2048; % nfft for plotting section
N=10; % number of digital elements in array.  Number of adaptive degrees of freedom (DOFs) or input channels where N > 1
Kmax=floor(6*N);% Set number of realizations (ie samples to simulate) to create of training-data vectors aka "snapshots" 
% snapshot just means an instant in time, for STAP this is fast-time, 
% beamforming theory (RMB rule) says you need (2*N)-3 to get within 3 dB of optimal SINR on avg, this DOES assume gaussian
% some folks call it the "2N" rule.
% After you hit 10*N you're going to have the bulk of the performance, getting fractions of a dB after tha tpoint
% your samples need to be IID for that to be true, and the jammer must not have moved, else its stale weights, due to non-stationary data
% in radar you're always updating your R_est, and the rx pulsed should not be included in samples used to calc R_est
% dont put mikes name on it for now

% The adaptive solution's pattern will "calm down", as you add more samples. it will always have unity gain, but the sidelobes will come down
% for N=10 try doing a 20 samples vs 100 samples and looking at the sidelobe levels
% The quienscent pattern *is* the conventional beamformer (aka nonadaptive beamforming) when your weights = s, fix it earlier in the chapter
% when you take the fft to find the pattern, just call that calculating the beam pattern
% eg "adaptive beam pattern", "non-adaptive beam pattern"
% remmeber that you can calc the SINR_opt using the actual covariance estimate, its like an avg of avges

%%%%%%%%%%%%%%% Build jammer covariance matrix %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%		J= number of jammers
%		SJ2DB= J vector of jammer powers in db
%		PHIDJ= J vector of jammer angles in degrees
%		G= N vector gains in db
%		fb= the fractional bandwidth = bandwidth/center freq.

% make a note that multipath from the same jammer is not treated as 2 diff jammers, you would need a more sophistictead model
% for a fixed position phased array you would sweep look_direction
% we're not modeling any signal coming from the look direction
% but if there was a signal coming through from the look direction, it would be rx at unity gain
% for end-to-end sim you would add that SOI
% the point is we can predict the SINR without even modeling the SOI, we know its going to be unity gain
look_direction = 0; % Set array beam position: 0 = broadside (centered) or else +-(integer less than N/2) moves beam around
s=exp(1i*(2*pi/N)*look_direction*[0:N-1]).'; % Steering vector, aka array response to a signal comign from the look dir
% This is done here to make our life easier later when we add element gain
s=s/norm(s); % normalize steering vector - makes the expression s's = 1 mean that s has unit power per element (not unit energy in total) Thus, a*s has power per array element of p = |a*s|^2 = |a|^2 *1 = |a|^2


% Set # jammers and jammer parameters
% ULA specifics come into play here
% When fractional bandwidth gets too high, the ULA model will break down, avoid going above 0.1
% These jamming powers of 30 dB have to be relative to something, so its relative to thermal noise equal to 0 dB
J=3; SJ2DB=[30 30 30]; THETADJ= [-70 -20 40]; PHIDJ=(pi*sin(THETADJ*(pi/180)))*(180/pi); G=zeros(1,N); fb=0; % List jammer powers and angles in order from neg. to pos.
%J=5; SJ2DB=[30 10 30 50 30]; THETADJ= [-30 -20 -10 5 50]; PHIDJ=(pi*sin(THETADJ*(pi/180)))*(180/pi); G=zeros(1,N); fb=0; % List jammer powers and angles in order from neg. to pos. 

% Create jammer covariance matrix
Rx_array = Covmatrix2(N,J,SJ2DB,PHIDJ,G,fb);
%  Rx_array=eye(N); % uncomment this for zero jammers

% Calculate sqrt of covariance matrix used to 'color' random vectors to match jammer covariance matrix
A=Rx_array^(.5); % Effective Cholesky factoriz. of correlation matrix: produces the Hermitian of the result textbooks might produce
% there's a 2-line proof that explains why we need to do this
% you cuold do this as part of NonStatData but we'll just do it here beforehand
% same idea as the reason you multiply by the sqrt of variance when scaling a RV

% Make it clear to readers that this does not require creating the samples
% This is incredibly powerful, being able to predict perforamnce without even doing sample-level simulation
% This is the upper bound, best you can do, best avg SINR, doesnt assume guassian noise, can be any noise of that power
% its called "Optimal SINR", this is in textbooks
% This is relative to unity value of thermal noise per rx element (eg "1 watt per element assumption") you can pick whether its relative to dBW or dBm, you have to pick one
% this is the ktb noise
sinr_opt = real(s'*(inv(Rx_array))*s); % 'real' used to clean up (remove) neglible imaginary component caused by finite precision Covariance matrix inversion 
% essentially the mag squared of S^H * sqrt(R^-1), its just calculating a power, relative to our unity noise
% this model/code assumes s is unity power (so is equal to the noise which is also unity gain)
% so waht we're really calculating here is the interference-to-noise realization, it's trivial to add in signal
% the point is real(s'*(inv(Rx_array))*s) is just capturing the interference-to-noise ratio even though it's officially called "Optimal SINR"
% "adaptive SINR" is just calculated from the R_est, same formula but it goes from opitmal to adaptive SINR
% if mike uses "ideal" is just means optimal

% Create correlated data matrix c1 which is generated from the jammer covariance matrix
% This function actually creates the stationary training data that is representaitive of the cov matrix was just made
% lets us go from white noise to correlated (aka colored) noise that represents the jamming scenario
% the +1 doesnt matter, can take it out
c1= NonStatData(A,N,Kmax+1,0); % dimensions are N x Kmax. There are Kmax independent vector realizations of a random vector having jammer covariance matrix

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Plots  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Plot optimal and adaptive antenna patterns
non_adaptive=20*log10(fftshift(abs(fft(s,nfft))/sqrt(N))); % used '/sqrt(N)' to force peak to zero since fft kernel is not normalized to unit norm (see below)
w_ideal = (inv(Rx_array)*s)/(s'*(inv(Rx_array))*s);
IDEAL = 20*log10(fftshift(abs(fft(w_ideal,nfft))/sqrt(N))); % Divide by sqrt(N) since fft kernel is not normalized to unit norm like s is so dividing by sqrt(N) effectively normalized the fft kernel so the max output is 1 or 0 dB
faxis=get_fft_axis(nfft,2*pi,1); % since unambig range = -pi:pi (i.e., 2pi) sample rate is twice this or 4pi ?? rad/sample
faxis_physical_angle = asin(faxis/pi) * (180/pi);
h0=figure; set(h0,'DefaultLineLineWidth',2);
plot(faxis_physical_angle,non_adaptive, 'b'), hold on
plot(faxis_physical_angle,IDEAL, 'r')
ax1 = [-90 90 -80 10];
xlabel('Azimuth angle (deg)', 'FontWeight', 'bold', 'FontSize', Fsize), ylabel('Power, dB', 'FontWeight', 'bold', 'FontSize', Fsize)
title(['N=',num2str(N),' DOF, K=', num2str(Kmax), ', SINR_o_p_t= ',num2str(10*log10(sinr_opt))], 'FontWeight', 'bold', 'FontSize', Fsize)
axis(ax1);
set(h0,'DefaultLineLineWidth',1);
for ii=1: length(THETADJ)
    plot([THETADJ(ii) THETADJ(ii)], [ax1(3) ax1(4)],'k')
end
legend('Non-Adaptive Pattern','Optimal Pattern', ...
    ['Jammer locations: JNR = ', num2str(SJ2DB),' dB  (L to R),  Fractional BW = ', num2str(fb)], ...
    'Location', 'SouthWest')
legend boxoff
set(gca, 'FontWeight', 'bold', 'FontSize', Fsize)
grid
