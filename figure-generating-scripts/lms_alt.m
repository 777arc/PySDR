clc
clear all;
close all;


Fc=4000000;   %Carrier Frequency =1Mhz
f=Fc;
Fs=16000000;  %sampling frequency
fm=700000; 
Fd=fm;
sensors=4; % number of Sensors 
angle_tx=pi/4;  %Transmitter angle = 45 degrees
angle_jam=pi/4; %jammer angle = 45 degree
sens_wts=[0.2 0.4 0.6 0.8]; %Assigning Sensor weights
c=3e08;     
lembda=c/f;    % Transmitted signal wavelength 

samps=2*(6*Fs/fm);       %   Number of samples of Data
max=(1/Fs)*(samps-1);              
t=0:1/Fs:(max);    

%Modulating Data
modsignal = sin(2*pi*fm*t);           % Baseband Signal
modsignal(modsignal>0)=1;  
modsignal(modsignal<0)=-1; 
CAR=sin(2*pi*Fc*t); %carrier
t_sig=modsignal.*CAR;                 % modulated Data
ch_samps=length(t_sig);               %Modulated Signal Data Samples
g=0:ch_samps-1;
% figure(2);                            
% plot(g,t_sig);                        %plot Transmitted signal
% title('Transmitted Modulated signal'); grid on;
% axis([0 8e6 0 1]);

d=lembda/2;  %Sensor separation
meu=15e-6;   %Step size

%FFT of Transmitted signal
k=ch_samps;
fft_samps = 2^nextpow2(k); 
t_fft = fft(t_sig,fft_samps)/k;
fprime = Fs/2*linspace(0,1,fft_samps/2);
figure(3); 
subplot(3,1,1)             %plot frequency components of Transmitted Signal
plot(fprime,2*abs(t_fft(1:fft_samps/2))); 
%grid on;
title('Original Transmitted signal frequency spectrum');
xlabel('frequency Hz');
ylabel('magnitude');
axis([0 8e6 0 1])
%Jammer Signal 
t=0:1/Fs:max;
j_sig=1*sin(2*pi*Fc*t);

%FFT of Jamming Signal
fft_samps = 2^nextpow2(k); 
j_fft = fft(j_sig,fft_samps)/k;
fprime = Fs/2*linspace(0,1,fft_samps/2);
% figure(4);  
% plot(fprime,2*abs(j_fft(1:fft_samps/2)));   %plot frequency components  of Jamming Signal
% grid on;
% title('Spectrum of Jamming signal');
% xlabel('Frequency (Hz)');
% ylabel('magnitude');
% axis([0 8e6 0 1])


%Array Propogation Vectors
for t2=1:sensors;
 v(t2)=exp(i*(t2-1)*2*pi*d*sin(angle_tx)*1/lembda);%propagation vector 

end

for t3=1:sensors;
 eeta(t3)=exp(i*(t3-1)*2*pi*d*sin(angle_jam)*1/lembda);%Propagation Vector for Jamming Signal
end

%Jammer Reception at the Sensor Array
j_rcvd=j_sig'*eeta;   %Jamming signal Reception @ Sensors


% j_sig+t_sig = data after reception @ anteena
 x=t_sig'*v+j_rcvd;
 
%FFT of Recieved Signal @ Sensors
fft_samps = 2^nextpow2(k); 
x_fft = fft(x,fft_samps)/k;
fprime = Fs/2*linspace(0,1,fft_samps/2);
figure(3);  
subplot(3,1,2 )
plot(fprime,2*abs(x_fft(1:fft_samps/2))); % Plot frequency components of Rxd signal
%Grid on;
title('Signal After reception at Antenna (Jamming Signal + Desired Signal)');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
axis([0 8e6 0 1])

%LMS Algorithm
 for n1=1:ch_samps
x_est=sens_wts*x';
E=t_sig-x_est;
sens_wts=sens_wts+(meu*E*x);                 
 end

%FFT of Estimated
fft_samps = 2^nextpow2(k);
z_fft = fft(x_est,fft_samps)/k;
fprime = Fs/2*linspace(0,1,fft_samps/2);
figure(3);  
subplot(3,1,3); 
plot(fprime,2*abs(z_fft(1:fft_samps/2)));  % Plot frequency components of Estimated signal                     
%Grid on;
title('Estimated signal frequency spectrum');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
axis([0 8e6 0 1])
%**************************
%Scatter plot of estimated signal

% y_est = ddemodce(x_est,Fd,Fs*2,'psk',2);
% % y_est= circshift(y_ester,3)
% figure(1); 
% subplot(2,1,2); 
% stem(y_est,'filled'),grid on; 
% axis([2 101 0 1]);