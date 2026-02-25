function R= Covmatrix2(N,J,SJ2DB,PHIDJ,G,fb)

% THIS IS EXACTLY WHAT I IMPLEMENTED AS PART OF https://pysdr.org/content/doa.html#simulating-wideband-interferers

% ***wideband jammers***
% computes covariance matrix given jammer parameters
% this function computes the NxN covariance matrix,R, associated with a main antenna
% and N-1 aux elements 

%		N-1= number of aux elements
%		J= number of jammers
%		SJ2DB= J vector of jammer powers in db
%		PHIDJ= J vector of jammer angles in degrees
%		G= N vector gains in db
%		fb= the fractional bandwidth,bandwidth/center freq.

G=10.^(.1*G);  % Convert to numeric

for j=1:J
SJ2(j)= 10^(.1*SJ2DB(j));

% compute phi in radians

PHIRJ(j)= PHIDJ(j)*(pi/180);
end

% create NxJ angle/power array A

R(1:N,1:N)=0;

for m=1:N
   for n=1:N
      for j=1:J
         % Preserving phase. we arent here but you could have a complex number for G
         % this assumes the jammer is white noise
         % sinc term is for wideband (note when fb=0, sinc(...)=1 which models the narrowband situation)
         % last term is just the correct way for keeping track of phase to create this kind of covariance matrix
         % on the diagonals when m=n this goes to 1
         R(m,n)=R(m,n)+SJ2(j)*sqrt(G(m)*G(n))*(sinc(.5*fb*(m-n)*PHIRJ(j)/pi))*exp(i*(m-n)*PHIRJ(j));
      end
   end
end


% create cov matix
% This adds in the thermal noise of each element, because up until this point R is just jammer covariance matrix
R = eye(N) + R;
% at this point its the jammer + thermal noise covariance matrix
% this assumes "unity" noise power, when we defined our jammer power earlier that had to be relative to unity thermal noise of the rx's

