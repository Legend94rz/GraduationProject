function BSGD(dataset)
	global hash ds;
	hash = java.util.Hashtable;
	hash.put('svmguide1',0.01);hash.put('ijcnn1',0.05);hash.put('cbcl',5);hash.put('a7a',0.1);hash.put('w7a',0.2);
	ds=dataset;
	fprintf('*********BSGD Begin*********\n');
	load(strcat('data\',dataset,'.mat'));
	N=size(train,1);train = [train,ones(N,1)];
	[N,d] = size(train);
	
	[L,test] = readdata(dataset,'test');
	M=size(test,1);test = [test,ones(M,1)];
	[M,~] = size(test);
	
	BSGD_history = [];SV_history=[];alpha_history=[];
	nperpart =N/fold;
	B=ceil(d*1.4);
	for lambda=[0.0001,0.001,0.01,0.1,1,10,100,1000]
		fprintf('lambda: %d, Sigma: %d\n',lambda,hash.get(dataset));
		rule = (1:fold)';
		result=zeros(1,5);
		for TestRound=1:fold
			tmp = (rule(1)-1)*nperpart+1 : rule(1)*nperpart;
			[I,SV,alpha,beta] = Init(train(tmp,:),label(tmp),B,lambda);
			for partion = 1:fold-1
				iv = (rule(partion)-1)*nperpart+1 : rule(partion)*nperpart;
				[I,SV,alpha,beta] = Train(B,I,SV,alpha,beta,lambda,train( iv ,: ),label(iv) );
			end
			iv=(rule(fold)-1)*nperpart+1:rule(fold)*nperpart;
			aResult = Test(SV,alpha,train( iv ,:),label(iv) );
			result = (aResult+(TestRound-1)*result)/TestRound;
			fprintf(' tp:%.4f, fn:%.4f, fp:%.4f, tn:%.4f, P:%.4f\n',aResult(1),aResult(2),aResult(3),aResult(4),aResult(5) );
			rule=circshift(rule,1);
		end
		fprintf('average: tp:%.4f, fn:%.4f, fp:%.4f, tn:%.4f, P:%.4f\n',result(1),result(2),result(3),result(4),result(5) );
		BSGD_history = cat(1,BSGD_history,result);
		SV_history = cat(3,SV_history,full(SV));
		alpha_history = cat(1,alpha_history,alpha');
	end
	fprintf('summery on %s: \n',dataset);
	disp(BSGD_history);
	[~,~,location] = SelectOpt(BSGD_history);
	SV = SV_history(:,:,location);
	alpha = alpha_history(location,:)';
	finalResult = Test(SV,alpha,test,L);
	fprintf('final, optIndex: %d\n',location);
	disp(finalResult);
	SaveResult(dataset, finalResult);
	fprintf('*********BSGD End*********\n');
end
function [I,SV,alpha,beta]=Train(B,I,SV,alpha,beta,lambda,train,label)
	global hash ds;
	[N,~] = size(train);
	for it=B+1:N
		t = randi([1,N]);
		xt = train(t,:);yt = label(t);
		margin = yt*kernel(xt,SV,hash.get(ds))*alpha;
		if(margin<1)
			beta_ = 1/(lambda*it)*yt;
			p=randi([1,B]);
			I(p) = t;
			SV(p,:) = xt;
			beta(p) = beta_;
			alpha(p) = beta_*t/it;
		end
	end
end
function aResult = Test(SV,alpha,test,L)
	global hash ds;
	[M,~] = size(test);
	CM=zeros(2,2);
	PositiveSample = sum(L==1);
	NegtiveSample = sum(L==-1);
	for t = 1:M
		xt = test(t,:);yt=L(t);
		ey = sign( kernel(xt,SV,hash.get(ds))*alpha );
		CM((-1==yt)+1,(-1==ey)+1) = CM((-1==yt)+1,(-1==ey)+1)+1;
	end
	t=(CM(1,1)+CM(2,2))/(PositiveSample+NegtiveSample);
	CM(1,:) = CM(1,:)/PositiveSample;
	CM(2,:) = CM(2,:)/NegtiveSample;
	aResult = [reshape(CM,[1,4]),t];
end
function [I,SV,alpha,beta]=Init(train,label,B,lambda)
	SV=train(1:B,:);
	I=(1:B)';
	beta =label(1:B).*(1/lambda).*I;
	alpha=beta.*I/B;
end
function SaveResult(dataset,BSGD_OPT)
	filename=strcat(dataset,'.mat');
	if(exist(filename,'file'))
		save(filename,'BSGD_OPT','-append');
	else
		save(filename,'BSGD_OPT');
	end
end


%dataset = 'a7a';
function xBSGD(dataset)
fprintf('*********BSGD Begin*********\n');
[label,train] = readdata(dataset,'train');
N=size(train,1);train = [train,ones(N,1)];
[N,d] = size(train);

[L,test] = readdata(dataset,'test');
M=size(test,1);test = [test,ones(M,1)];
[M,~] = size(test);

BSGD_history = [];
%Pegasos learning rate : eta = 1/(lambda*t)
hash = java.util.Hashtable;
hash.put('svmguide1',0.01);hash.put('ijcnn1',0.05);hash.put('cbcl',5);hash.put('a7a',0.1);hash.put('w7a',0.2);
for lambda = [0.0001,0.001,0.01,0.1,1,10,100,1000]
B=500;
%%init: consider first B samples as Support Vectors
SV=train(1:B,:);
I=(1:B)';
beta =label(1:B).*(1/lambda).*I;
alpha=beta.*I/B;

fprintf('lambda: %d, Sigma: %d\n',lambda,hash.get(dataset));

for it=B+1:N
	t = randi([1,N]);
	xt = train(t,:);yt = label(t);
	margin = yt*kernel(xt,SV,hash.get(dataset))*alpha;
	if(margin<1)
		beta_ = 1/(lambda*it)*yt;
		p=randi([1,B]);
		I(p) = t;
		SV(p,:) = xt;
		beta(p) = beta_;
		alpha(p) = beta_*t/it;
	end
end
CM=zeros(2,2);
PositiveSample = sum(L==1);
NegtiveSample = sum(L==-1);
for t = 1:M
	xt = test(t,:);yt=L(t);
	ey = sign( kernel(xt,train(I,:),hash.get(dataset))*alpha );
	CM(int8(-1==yt)+1,int8(-1==ey)+1) = CM(int8(-1==yt)+1,int8(-1==ey)+1)+1;
end
BSGD_history = cat(1,BSGD_history,reshape(CM,[1,4]));
fprintf('Test. %d/%d\n',CM(1,1)+CM(2,2),M);
fprintf(' tp:%.4f, fn:%.4f, fp:%.4f, tn:%.4f\n',CM(1,1)/PositiveSample,CM(1,2)/PositiveSample,CM(2,1)/NegtiveSample,CM(2,2)/NegtiveSample);
end
filename=strcat(dataset,'.mat');
if(exist(filename,'file'))
	save(filename,'BSGD_history','-append');
else
	save(filename,'BSGD_history');
end
fprintf('*********BSGD End*********\n');
end