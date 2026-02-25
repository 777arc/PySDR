function R= Covmatrix2(N,J,SJ2DB,PHIDJ,G,fb)

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
         R(m,n)=R(m,n)+SJ2(j)*sqrt(G(m)*G(n))*(sinc(.5*fb*(m-n)*PHIRJ(j)/pi))*exp(i*(m-n)*PHIRJ(j));
      end
   end
end


% create cov matix

R= eye(N)+ R;



