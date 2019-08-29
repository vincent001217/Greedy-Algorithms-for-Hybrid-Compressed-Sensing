% Paper: Greedy Algorithms for Hybrid Compressed Sensing
% Experiment 2
% Code Author: Ching-Lun Tai, vincent001217@gmail.com
% Updated Date: 2019.6.24
% All Rights Reserved

clear all
N = 256;
Pow_min = 2;
Pow_max = 5;
Pow_range = Pow_max-Pow_min;
MC_range = 500;
SNR = 0;         % 0,10

%%% Proposed %%%
M = 48;           % real
MQ = 32*16;        % one-bit
rec_cor2 = zeros(Pow_range+1,1);
rec_err2 = zeros(Pow_range+1,1);
rec_cor3 = zeros(Pow_range+1,1);
rec_err3 = zeros(Pow_range+1,1);
%%%%%%%%%%%%%%%%

%%% OMP & CoSaMP & SP %%%
MR = 64;          % real
rec_corO = zeros(Pow_range+1,1);
rec_errO = zeros(Pow_range+1,1);
rec_corC = zeros(Pow_range+1,1);
rec_errC = zeros(Pow_range+1,1);
rec_corS = zeros(Pow_range+1,1);
rec_errS = zeros(Pow_range+1,1);
%%%%%%%%%%%%%%%%%%%%%%%%
h = waitbar(0,'please wait');

for K = 2.^[Pow_min:Pow_max]
    str = ['Progressing...',num2str((log2(K)-Pow_min+1)/(Pow_range+1)*100),'%'];
    waitbar((log2(K)-Pow_min+1)/(Pow_range+1),h,str)
    counter2 = 0;
    counter3 = 0;
    counter_O = 0;
    counter_C = 0;
    counter_S = 0;
    
    counter_e2 = 0;
    counter_e3 = 0;
    counter_O_e = 0;
    counter_C_e = 0;
    counter_S_e = 0;
    for MC = 1:MC_range
        supp = sort(randperm(N,K),'ascend'); % correct support answer
        x = zeros(N,1);
        x(supp) = randn(K,1);                % Gaussian r.v.
        A = randn(M,N)/sqrt(M);              % Proposed, real
        AQ = randn(MQ,N)/sqrt(MQ);           % Proposed, one-bit
        AR = randn(MR,N)/sqrt(MR);           % real measurements
        noise = randn(N,1);
        x_tilde = x+noise/norm(noise)*norm(x)/(10^(SNR/20));
        y = A*x_tilde;
        yQ = AQ*x_tilde;
        yR = AR*x_tilde;

        [xj_hat2,Omega2] = Proposed_1(y,yQ,A,AQ,K);
        [xj_hat3,Omega3] = Proposed_2(y,yQ,A,AQ,K,Omega2);
        [xO_hat,r1,normR1,residHist1, errHist1] = OMP(AR,yR,K,[],[]);
        [xC_hat,r2,normR2,residHist2, errHist2] = CoSaMP(AR,yR,K,[],[]);
        Rec = CSRec_SP(K,AR,yR);
        xS_hat = Rec.x_hat;
        
        counter2 = counter2+length(intersect(Omega2,supp));
        counter3 = counter3+length(intersect(Omega3,supp));
        counter_O = counter_O+length(intersect(find(xO_hat),supp));
        counter_C = counter_C+length(intersect(find(xC_hat),supp));
        counter_S = counter_S+length(intersect(find(xS_hat),supp));
        
        counter_e2 = counter_e2+norm(x)^2/(norm(x-xj_hat2)^2);
        counter_e3 = counter_e3+norm(x)^2/(norm(x-xj_hat3)^2);
        counter_O_e = counter_O_e+norm(x)^2/(norm(x-xO_hat)^2);
        counter_C_e = counter_C_e+norm(x)^2/(norm(x-xC_hat)^2);
        counter_S_e = counter_S_e+norm(x)^2/(norm(x-xS_hat)^2);
    end
    rec_cor2(log2(K)-Pow_min+1) = counter2/(MC_range*K);
    rec_cor3(log2(K)-Pow_min+1) = counter3/(MC_range*K);
    rec_corO(log2(K)-Pow_min+1) = counter_O/(MC_range*K);
    rec_corC(log2(K)-Pow_min+1) = counter_C/(MC_range*K);
    rec_corS(log2(K)-Pow_min+1) = counter_S/(MC_range*K);
    
    rec_err2(log2(K)-Pow_min+1) = counter_e2/MC_range;
    rec_err3(log2(K)-Pow_min+1) = counter_e3/MC_range;
    rec_errO(log2(K)-Pow_min+1) = counter_O_e/MC_range;
    rec_errC(log2(K)-Pow_min+1) = counter_C_e/MC_range;
    rec_errS(log2(K)-Pow_min+1) = counter_S_e/MC_range;
end
close(h);

figure(1)
plot(2.^[Pow_min:Pow_max],rec_cor2,'b-+')
hold on
plot(2.^[Pow_min:Pow_max],rec_cor3,'b-x')
hold on
plot(2.^[Pow_min:Pow_max],rec_corO,'g--*')
hold on
plot(2.^[Pow_min:Pow_max],rec_corC,'k--^')
hold on
plot(2.^[Pow_min:Pow_max],rec_corS,'m--square')
grid on
xlabel('Sparsity')
ylabel('Correct rate')
legend('Algotithm 1','Algorithm 2','OMP [20]','CoSaMP [23]','SP [24]')
axis([2^Pow_min,2^Pow_max,-inf,inf]);

figure(2)
plot(2.^[Pow_min:Pow_max],10*log10(rec_err2),'b-+')
hold on
plot(2.^[Pow_min:Pow_max],10*log10(rec_err3),'b-x')
hold on
plot(2.^[Pow_min:Pow_max],10*log10(rec_errO),'g--*')
hold on
plot(2.^[Pow_min:Pow_max],10*log10(rec_errC),'k--^')
hold on
plot(2.^[Pow_min:Pow_max],10*log10(rec_errS),'m--square')
grid on
xlabel('Sparsity')
ylabel('Recovery SNR \xi_r (dB)')
legend('Algotithm 1','Algorithm 2','OMP [20]','CoSaMP [23]','SP [24]')
axis([2^Pow_min,2^Pow_max,-inf,inf]);