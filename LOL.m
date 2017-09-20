function LOL(dataset)
	fprintf('*********LOL Begin*********\n');
	%{
	load(strcat('data\',dataset,'.mat'));
	N=size(train,1);train = [train,ones(N,1)];
	[N,d] = size(train);
	
	[L,test]=readdata(dataset,'test');
	M=size(test,1);test = [test,ones(M,1)];
	[M,~] = size(test);
	%}
	load(strcat('data\',dataset,'.mat'));
	N = size(Data,1);Data=[Data,ones(N,1)];d=size(Data,2);
	p = ceil(N*0.8);
	train = Data(1:p,:);	label = Label(1:p,:);
	test = Data(p+1:end,:);	L = Label(p+1:end,:);
	fold=5;
	
	LOL_history = [];W_history=[];count_history=[];P_history=[];
	k=100;lambda=1;nperpart = fix(p/fold);
	for C=[1]
		fprintf('======C: %d==========\n',C);
		rule = (1:fold)';
		result=zeros(1,5);
		for TestRound=1:fold
			[W,count,P] = Init(d,k);
			for partion = 1:fold-1
				iv = (rule(partion)-1)*nperpart+1 : rule(partion)*nperpart;
				[W,count,P]=Train(W,count,P,k,lambda,C,train(iv ,: ),label(iv) );
			end
			iv=(rule(fold)-1)*nperpart+1:rule(fold)*nperpart;
			aResult = Test( W,P,k,train( iv ,:),label(iv) );
			result = (aResult+(TestRound-1)*result)/TestRound;
			fprintf(' tp:%.4f, fn:%.4f, fp:%.4f, tn:%.4f, P:%.4f\n',aResult(1),aResult(2),aResult(3),aResult(4),aResult(5) );
			rule=circshift(rule,1);
		end
		fprintf('average: tp:%.4f, fn:%.4f, fp:%.4f, tn:%.4f, P:%.4f\n',result(1),result(2),result(3),result(4),result(5) );
		LOL_history = cat(1,LOL_history,result);
		W_history = cat(3,W_history,W);
		count_history = cat(1,count_history,count);
		P_history=cat(3,P_history,P);
	end
	fprintf('summery on %s: \n',dataset);
	disp(LOL_history);
	[~,~,location] = SelectOpt(LOL_history);
	W = W_history(:,:,location);
	count=count_history(location,:);
	P = P_history(:,:,location);
	finalResult = Test(W,P,k,test,L);
	fprintf('final, optIndex: %d\n',location);
	disp(finalResult);
	SaveResult(dataset, finalResult);
	fprintf('*********LOL End*********\n');
end
function [W,count,P]=Train(W,count,P,k,lambda,C,train,label)
	[N,d] = size(train);
	for t=1:N
		xt = train(t,:);
		i=t;
		if(t<=k)
			P(t,:) = xt;
			count(t)=count(t)+1;
		else
			xtExt = repmat(xt,k,1);
			[~,i] = min( sum( (xtExt-P).^2 ,2 ) );
			P(i,:) = P(i,:)+1/count(i)*(xt-P(i,:));
			count(i) = count(i)+1;
		end
		Xt = zeros(k+1,d);
		Xt(1,:) = xt./sqrt(lambda);
		Xt(i,:) = xt;
		result = Xt*W;
		yt = label(t);
		loss = max(0,1-yt*sum(result(:)) );
		tmp = Xt.^2;
		yita = min(C,loss/sum(tmp(:)));
		W = W + yita * yt * Xt';
	end
end
function aResult = Test(W,P,k,test,L)
	[M,d] = size(test);
	CM = zeros(2,2);
	PositiveSample = sum(L==1);
	NegtiveSample = sum(L==-1);
	for t = 1:M
		xt = test(t,:);
		Xt = zeros(k+1,d);
		Xt(1,:) = xt;
		xtExt = repmat(xt,k,1);
		[~,i] = min( sum( (xtExt-P).^2 ,2 ) );
		Xt(i,:) = xt;
		result = Xt*W;
		ey=sign( sum( result(:) ) );
		yt=L(t);
		CM((-1==yt)+1,(-1==ey)+1) = CM((-1==yt)+1,(-1==ey)+1)+1;
	end
	t=(CM(1,1)+CM(2,2))/(PositiveSample+NegtiveSample);
	CM(1,:) = CM(1,:)/PositiveSample;
	CM(2,:) = CM(2,:)/NegtiveSample;
	aResult = [reshape(CM,[1,4]),t];
end
function [W,count,P]=Init(d,k)
	W = zeros(d,k+1);
	count = zeros(1,k);
	P = zeros(k,d);
end
function SaveResult(dataset,LOL_OPT)
	filename=strcat(dataset,'.mat');
	if(exist(filename,'file'))
		save(filename,'LOL_OPT','-append');
	else
		save(filename,'LOL_OPT');
	end
end


function xLOL(dataset)
fprintf('*********LOL Begin*********\n');

[label,train] = readdata(dataset,'train');
N=size(train,1);train = [train,ones(N,1)];
[N,d] = size(train);

[L,test]=readdata(dataset,'test');
M=size(test,1);test = [test,ones(M,1)];
[M,~] = size(test);

k=60;lambda=1;

W = zeros(d,k+1);
count = zeros(1,k);
P = zeros(k,d);

LOL_history = [];
for Ci=0
C=10^Ci;
fprintf('C: %d\n',C);
for t=1:N
	xt = train(t,:);
	i=t;
	if(t<=k)
		P(t,:) = xt;
		count(t)=count(t)+1;
	else
		xtExt = repmat(xt,k,1);
		[~,i] = min( sum( (xtExt-P).^2 ,2 ) );
		P(i,:) = P(i,:)+1/count(i)*(xt-P(i,:));
		count(i) = count(i)+1;
	end
	Xt = zeros(k+1,d);
	Xt(1,:) = xt./sqrt(lambda);
	Xt(i,:) = xt;
	result = Xt*W;
	yt = label(t);
	loss = max(0,1-yt*sum(result(:)) );
	tmp = Xt.^2;
	yita = min(C,loss/sum(tmp(:)));
	W = W + yita * yt * Xt';
end
CM = zeros(2,2);
PositiveSample = sum(L==1);
NegtiveSample = sum(L==-1);
for t = 1:M
	xt = test(t,:);
	Xt = zeros(k+1,d);
	Xt(1,:) = xt;
	[~,i] = min( sum( repmat(xt,k,1) ,2 ) );
	Xt(i,:) = xt;
	result = Xt*W;
	ey=sign( sum( result(:) ) );
	yt=L(t);
	CM(int8(-1==yt)+1,int8(-1==ey)+1) = CM(int8(-1==yt)+1,int8(-1==ey)+1)+1;
end
LOL_history = cat(1,LOL_history,reshape(CM,[1,4]));
fprintf('Test. %d/%d\n', CM(1,1)+CM(2,2),M);
fprintf(' tp:%.4f, fn:%.4f, fp:%.4f, tn:%.4f\n',CM(1,1)/PositiveSample,CM(1,2)/PositiveSample,CM(2,1)/NegtiveSample,CM(2,2)/NegtiveSample);
filename=strcat(dataset,'.mat');
if(exist(filename,'file'))
	save(filename,'LOL_history','-append');
else
	save(filename,'LOL_history');
end
end
fprintf('*********LOL End*********\n');
end