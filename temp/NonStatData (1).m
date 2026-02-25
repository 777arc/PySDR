function z= NonStatData(A,N,K,agam);

% this function generates an NxK data matrix z
% for a canceller given the NxN cov matrix R , chol(R)=A
% normalized so that internal noise power = 1
% sample vectors are sphereically invariant generated from a gamma function with a given 'agam' parameter 
% mean of gamma =agam*bgam =1 and variance of gamma= agam*bgam^2. If agam=0, then no nonstationarity

a=sqrt(2);
A=A/a;
if agam==0
    c=1;
else
bgam=1/agam;
end

for k=1:K-1
 
    if agam>0 
        c=gamrnd(agam,bgam);
    end
    
z(1:N,k)= c*A'*(randn(N,1)+i*randn(N,1));
end
z(1:N,K)= A'*(randn(N,1)+i*randn(N,1));

