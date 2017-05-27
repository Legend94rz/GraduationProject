function Pegasos(dataset)
	global K;
	K = 1;		% the norm of the subset of training set. When k==1, the algorithm is a kind of stochastic gradient descent method.
	fprintf('*********Pegasos Begin*********\n');
	load(strcat('data\',dataset,'.mat'));
	N = size(train,1);train = [train,ones(N,1)];
	[N,d]=size(train);
	
	[L,test] = readdata(dataset,'test');
	M=size(test,1);test = [test,ones(M,1)];
	
	Pegasos_history = [];w_history=[];nperpart =N/fold;
	for lambdaK = -5:5
		lambda = 10^lambdaK;		% 1/lambda === C
		fprintf('======lambda: %d==========\n',lambda);
		rule = (1:fold)';
		result=zeros(1,5);
		for TestRound=1:fold
			[w] = Init(d);
			for partion = 1:fold-1
				iv = (rule(partion)-1)*nperpart+1 : rule(partion)*nperpart;
				w=Train(lambda,w,train( iv ,: ),label(iv) );
			end
			iv=(rule(fold)-1)*nperpart+1:rule(fold)*nperpart;
			aResult = Test( w,train( iv ,:),label(iv) );
			result = (aResult+(TestRound-1)*result)/TestRound;
			fprintf(' tp:%.4f, fn:%.4f, fp:%.4f, tn:%.4f, P:%.4f\n',aResult(1),aResult(2),aResult(3),aResult(4),aResult(5) );
			rule=circshift(rule,1);
		end
		fprintf('average: tp:%.4f, fn:%.4f, fp:%.4f, tn:%.4f, P:%.4f\n',result(1),result(2),result(3),result(4),result(5) );
		Pegasos_history = cat(1,Pegasos_history,result);
		w_history = cat(1,w_history,w');
	end
	fprintf('summery on %s: \n',dataset);
	disp(Pegasos_history);
	[~,~,location] = SelectOpt(Pegasos_history);
	w = w_history(location,:)';
	finalResult = Test(w,test,L);
	fprintf('final, optIndex: %d\n',location);
	disp(finalResult);
	SaveResult(dataset, finalResult);
	fprintf('*********Pegasos End*********\n');

end
function w=Init(d)
	w = zeros(d,1);
end
function [w]=Train(lambda,w,train,label)
	global K;
	[N,d] = size(train);
	T=N;
	for it = 1:T
		Index = randperm(N,K);yita = 1/(lambda*it);
		A = train( Index,:);Ry = label(Index);
		E = sign(A*w);
		Ap = A(E~=Ry,:);
		Lp = Ry(E~=Ry);
		if(~isempty(Lp))
			w = (1-yita*lambda)*w+yita/K * sum( repmat(Lp,1,d).*Ap )';
		else
			w = (1-yita*lambda)*w;
		end
		%w = min(1,(1/sqrt(lambda))/sqrt(sum(w.^2)))*w;		%Optional,projection，没有显著差别%
	end
end
function aResult = Test(w,test,L)
	CM = zeros(2,2);
	[M,~] = size(test);
	PositiveSample = sum(L==1);
	NegtiveSample = sum(L==-1);
	for t = 1:M
		xt = test(t,:);
		ey = sign(xt*w);
		yt = L(t);
		CM((-1==yt)+1,(-1==ey)+1) = CM((-1==yt)+1,(-1==ey)+1)+1;
	end
	t=(CM(1,1)+CM(2,2))/(PositiveSample+NegtiveSample);
	CM(1,:) = CM(1,:)/PositiveSample;
	CM(2,:) = CM(2,:)/NegtiveSample;
	aResult = [reshape(CM,[1,4]),t];
end
function SaveResult(dataset,Pegasos_single_OPT)
	filename=strcat(dataset,'.mat');
	if(exist(filename,'file'))
		save(filename,'Pegasos_single_OPT','-append');
	else
		save(filename,'Pegasos_single_OPT');
	end
end

function xPegasos(dataset)
fprintf('*********Pegasos Begin*********\n');
[label,train] = readdata(dataset,'train');
N = size(train,1);train = [train,ones(N,1)];
[N,d]=size(train);

[L,test] = readdata(dataset,'test');
M=size(test,1);test = [test,ones(M,1)];

Pegasos_history = [];

T = N;		% the number of iteration
K = 10;		% the norm of the subset of training set. When k==1, the algorithm is a kind of stochastic gradient descent method.
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
	if(~isempty(Lp))
		w = (1-yita*lambda)*w+yita/K * sum( repmat(Lp,1,d).*Ap )';
	else
		w = (1-yita*lambda)*w;
	end
	%w = min(1,(1/sqrt(lambda))/sqrt(sum(w.^2)))*w;		%Optional,projection，没有显著差别%
end
CM = zeros(2,2);
PositiveSample = sum(L==1);
NegtiveSample = sum(L==-1);
for t = 1:M
	xt = test(t,:);
	ey = sign(xt*w);
	yt = L(t);
	CM(int8(-1==yt)+1,int8(-1==ey)+1) = CM(int8(-1==yt)+1,int8(-1==ey)+1)+1;
end
Pegasos_history = cat(1,Pegasos_history,reshape(CM,[1,4]));
fprintf('Test. %d/%d\n',CM(1,1)+CM(2,2),M);
fprintf(' tp:%.4f, fn:%.4f, fp:%.4f, tn:%.4f\n',CM(1,1)/PositiveSample,CM(1,2)/PositiveSample,CM(2,1)/NegtiveSample,CM(2,2)/NegtiveSample);
end
filename=strcat(dataset,'.mat');
if(exist(filename,'file'))
	save(filename,'Pegasos_history','-append');
else
	save(filename,'Pegasos_history');
end
fprintf('*********Pegasos End*********\n');
end
