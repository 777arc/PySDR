clear;

%DOA_1 = 1.0485
%DOA_2 = 2.0931
%DOA_error = 1.3024e-03

%%
c = 3e8; % Speed of light

M = 16; % Number of antennas in ULA
A = 2; % Amplitude of the desired signal
f = 2.4e9; % frequency in Hz
lambda = c/f; % Wavelength of desired signal
d = lambda/2; % Distance between antennas in meters (m)
theta = pi/3; % Desired incoming wave front DOA
tau = (d/c)*sin(theta); % Time delay between recived signal

Pn = .1; % noise factor

% Interfearing signal variables
theta_i = pi/4;
tau_i = (d/c)*sin(theta_i);
A_i = 0; % amplitude of interference
f_i = 1e9;

% complex adjustment weights vector initialization
w = zeros(M,1); 
w(1) = 1;
for n = 2:M
    w(n) = 1;
end
%w = rand(M,1);

ts = 1/(500*f); % sample time
t = 0:ts:(1/f)*6; % adaptation time

X = zeros(M,1);% incoming signal vector initialization
Error = zeros(1,size(t,2)); % Error signal vector initialization
D = zeros(1,size(t,2));% Desired signal vector initialization
Y = zeros(1,size(t,2));% recived signal vector initialization
N = zeros(1,size(t,2));% noise signal vector initialization
I = zeros(1,size(t,2));

draw_dec_count = 1;
frame_count = 1;
polfig = figure;
%polfig.Visible = 'off';
loops = floor(length(t)/10);
pol_movie(loops) = struct('cdata',[],'colormap',[]);

%run LMS algorithem
mu = .04;% LMS step size

for tl = 1:size(t,2)
    D(tl) = cos(2*pi*f*tl*ts); % Reference Signal
    N(tl) = Pn*randn(); % AWGN
    for n = 1:M
        % Incoming signals
        X(n,1) = A*cos(2*pi*f*ts*tl + 2*pi*f*(n-1)*tau) + N(tl) + A_i*cos(2*pi*f_i*ts*tl + 2*pi*f_i*(n-1)*tau_i);
    end
    S = w.*hilbert(real(X));
    Y(tl) = sum(S);
    Error(tl) = conj(D(tl)) - Y(tl);
    w(:) = w(:) + mu*X*Error(tl);% next weight calculation
    
    % Array Response
    if draw_dec_count > 10
        draw_dec_count = 0;
        
        omega = 0:0.0001:2*pi;
        H = 0;
        for m = 1:M
            H = H + w(m)*exp(-1i*(m-1)*(d/c)*sin(omega)*2*pi*f);
        end
        mag_H = abs(H);
        [mm, i] = max(mag_H);
    
        % draw frame
        polar(omega,mag_H);
        drawnow
        pol_movie(frame_count) = getframe;
        frame_count = frame_count + 1;
    else
        draw_dec_count = draw_dec_count + 1;
    end
       
end

%olfig.Visible = 'on';
%movie(pol_movie);

%%
% plot adaptation process
figure;
plot(t, real(Error));
title('LMS Error over time');
xlabel('Time (s)');
ylabel('LMS Error');

figure;
plot(t, D, t, real(Y));
title('Beamforming Output Durring Adaptation Process');
legend('Reference Signal', 'Received Signal');
xlabel('Time (s)');
ylabel('Signal Amplitude');

% Signal to interference plus noise ratio
%SINR = snr(Y,N + I)

% LMS iteration length
%iteration_length = size(t,2)

% Calculate different approximated DOA
DOA_1 = i*.0001
DOA_2 = abs(pi - DOA_1)

% Calculate the DOA error
DOA_error = abs(min([DOA_1- theta, DOA_2- theta]))

% Plot radiation patterns
figure;
polar(omega,mag_H);
title('ULA Radiation Pattern Polar');
xlabel('Degrees');
ylabel('Magnitude');

figure;
plot(omega,mag_H);
title('ULA Radiation Pattern Cartesian');
xlabel('Radians');
ylabel('Magnitude');

mag_H2 = 10*log(abs(H));
mag_H3 = mag_H2 - max(mag_H2);
figure;
plot(omega,mag_H3);
title('Normalized ULA Radiation Pattern');
xlabel('Radians');
ylabel('Db');