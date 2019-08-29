% Paper: Greedy Algorithms for Hybrid Compressed Sensing
% Algorithm 1
% Code Author: Ching-Lun Tai, vincent001217@gmail.com
% Updated Date: 2019.6.24
% All Rights Reserved

function [x_hat,Omega] = Proposed_OMP2(y,yQ,A,AQ,K)
N = size(A,2);
r = y;
Omega = [];
for num = 1:K
    beta = floor((K-num+1)/K*N);
    A_rep = A;
    A_rep(:,Omega) = zeros(size(A,1),length(Omega));
    [value,index] = sort(abs(A_rep'*r),'descend');
    Candid = index(1:beta);
    rec_x = zeros(N,length(Candid));
    for b = 1:length(Candid)
        Omega2 = sort([Omega,Candid(b)]);
        xj2 = A(:,Omega2)\y;
        rec_x(Omega2,b) = xj2;
    end
    tmp = repmat(yQ,1,length(Candid)).*(AQ*rec_x);
    [value2,index2] = max(sum(tmp >= 0));
    Omega = sort([Omega,Candid(index2)]);
    xj = rec_x(Omega,index2);
    r = y-A(:,Omega)*xj;
end
x_hat = zeros(N,1);
x_hat(Omega) = xj;
end