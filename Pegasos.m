% Mini-Batch Iterations without Projection
function [Pegasos_OPTCM]=Pegasos(dataset,Epsilon)
fprintf('*********Pegasos Begin*********\n');
[label,train] = readdata(dataset,'train');
N = size(train,1);train = [train,ones(N,1)];
[N,d]=size(train);

[L,test] = readdata(dataset,'test');
M=size(test,1);test = [test,ones(M,1)];

Pegasos_OPTCM = zeros(2,2);
Pegasos_history = [];
maxCorrect=0;

T = N;		% the number of iteration
K = 50;		% the norm of the subset of training set. When k==1, the algorithm is a kind of stochastic gradient descent method.
for lambdaK = -5:5	
lambda = 10^lambdaK;		% 1/lambda === C
fprintf('======lambda: %d==========\n',lambda);
w = zeros(d,1);
for it = 1:T
	Index = randperm(N,K);yita = 1/(lambda*it);
	A = train( Index,:);Ry = label(Index);
	E = sign(A*w);
	Ap = A(E~=Ry,:);
	Lp = Ry(E~=Ry);
	w = (1-yita*lambda)*w+yita/K * sum( repmat(Lp,1,d).*Ap )';
	%w = min(1,(1/sqrt(lambda))/sqrt(sum(w.^2)))*w;		%Optional,projection£¬Ã»ÓÐÏÔÖø²î±ð%
end
CM = zeros(2,2);
PositiveSample = sum(L==1);
NegtiveSample = sum(L==-1);
for t = 1:M
	xt = test(t,:);
	ey = sign(xt*w);
	yt = L(t);
	CM(int8(-1==yt)+1,int8(-1==ey)+1) = CM(int8(-1==yt)+1,int8(-1==ey)+1)+1;
	if(CM(1,1)+CM(2,2)>maxCorrect && CM(2,2)<=Epsilon*NegtiveSample && CM(1,1) <=Epsilon*PositiveSample)
		maxCorrect = CM(1,1)+CM(2,2);
		Pegasos_OPTCM = CM;
	end
end
Pegasos_history = cat(1,Pegasos_history,reshape(CM,[1,4]));
fprintf('Test. %d/%d\n',CM(1,1)+CM(2,2),M);
fprintf(' tp:%.4f, fn:%.4f, fp:%.4f, tn:%.4f\n',CM(1,1)/PositiveSample,CM(1,2)/PositiveSample,CM(2,1)/NegtiveSample,CM(2,2)/NegtiveSample);
end
Pegasos_OPTCM(1,:)=Pegasos_OPTCM(1,:)./sum(L==1);
Pegasos_OPTCM(2,:)=Pegasos_OPTCM(2,:)./sum(L==-1);
filename=strcat(dataset,'.mat');
if(exist(filename,'file'))
	save(filename,'Pegasos_OPTCM','-append');
else
	save(filename,'Pegasos_OPTCM');
end
save(filename,'Pegasos_history','-append');
fprintf('*********Pegasos End*********\n');
end
