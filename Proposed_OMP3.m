% Paper: Greedy Algorithms for Hybrid Compressed Sensing
% Algorithm 2
% Code Author: Ching-Lun Tai, vincent001217@gmail.com
% Updated Date: 2019.6.24
% All Rights Reserved

function [x_hat,Omega] = Proposed_OMP3(y,yQ,A,AQ,K,Omega0)
N = size(A,2);
Omega = Omega0;
Omega_old = [];
counter = 1;
Original = [1:N];
while 1
    if counter >= 2
        Omega_old = Omega;
    end
    Candid = setdiff(Original,Omega);
    rec_x = zeros(N,length(Candid));
    for b = 1:length(Candid)
        Omega2 = sort([Omega,Candid(b)]);
        xj2 = A(:,Omega2)\y;
        rec_x(Omega2,b) = xj2;
    end
    tmp = repmat(yQ,1,length(Candid)).*(AQ*rec_x);
    [value2,index2] = max(sum(tmp >= 0));
    Omega = sort([Omega,Candid(index2)]);
    Candid_matrix = nchoosek(Omega,K);
    rec_x2 = zeros(N,K+1);
    for i = 1:(K+1)
        xj3 = A(:,Candid_matrix(i,:))\y;
        rec_x2(Candid_matrix(i,:),i) = xj3;
    end
    tmp2 = repmat(yQ,1,K+1).*(AQ*rec_x2);
    [value3,index3] = max(sum(tmp2 >= 0));
    Omega = Candid_matrix(index3,:);
    if isequal(Omega,Omega_old)
        break;
    end
    counter = counter+1;
end
x_hat = zeros(N,1);
x_hat(Omega) = A(:,Omega)\y;
end