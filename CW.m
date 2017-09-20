function CW(dataset)
	fprintf('*********CW Begin*********\n');
	[L,test] = readdata(dataset,'test');
	M=size(test,1);test = [test,ones(M,1)];
	[M,~] = size(test);
	
	load(strcat('data\',dataset,'.mat'));
	N=size(train,1);train = [train,ones(N,1)];
	[N,d] = size(train);
	CW_history = [];mu_history=[];sigma_history=[];
	nperpart =N/fold;
	for yita=0.7:0.05:0.95
		fprintf('======yita: %d==========\n',yita);
		rule = (1:fold)';
		result=zeros(1,5);
		for TestRound=1:fold
			[mu,sigma,CM] = Init(d);
			for partion = 1:fold-1
				iv = (rule(partion)-1)*nperpart+1 : rule(partion)*nperpart;
				[mu,sigma]=Train(yita,mu,sigma,CM,train( iv ,: ),label(iv) );
			end
			iv=(rule(fold)-1)*nperpart+1:rule(fold)*nperpart;
			aResult = Test( mu,sigma,train( iv ,:),label(iv) );
			result = (aResult+(TestRound-1)*result)/TestRound;
			fprintf(' tp:%.4f, fn:%.4f, fp:%.4f, tn:%.4f, P:%.4f\n',aResult(1),aResult(2),aResult(3),aResult(4),aResult(5) );
			rule=circshift(rule,1);
		end
		fprintf('average: tp:%.4f, fn:%.4f, fp:%.4f, tn:%.4f, P:%.4f\n',result(1),result(2),result(3),result(4),result(5) );
		CW_history = cat(1,CW_history,result);
		mu_history = cat(1,mu_history,mu');
		sigma_history = cat(3,sigma_history,sigma);
	end
	fprintf('summery: \n');
	disp(CW_history);
	[~,~,location] = SelectOpt(CW_history);
	mu = mu_history(location,:)';
	sigma=sigma_history(:,:,location);
	finalResult = Test(mu,sigma,test,L);
	fprintf('final, optIndex: %d\n',location);
	disp(finalResult);
	SaveResult(dataset, finalResult);
	fprintf('*********CW End*********\n');
end
function [mu,Sigma]=Train(yita,mu,Sigma,CM,train,label)
	phi=norminv(yita,0,1);sigmav = inv(Sigma);
	[N,~]=size(train);
	for t = 1:N
		xt = train(t,:);
		yt = label(t);
		w = mvnrnd(mu,Sigma)';
		ey = sign(xt*w);
		Mi = yt*(xt*mu);
		Vi = xt*Sigma*xt';
		gamma = ( -(1+2*phi*Mi) + sqrt( (1+2*phi*Mi)^2-8*phi*(Mi-phi*Vi) ) )/(4*phi*Vi);
		alpha = max(gamma,0);
		
		mu = mu+alpha*yt*Sigma*xt';
		sigmav = sigmav + 2*alpha*phi*diag(xt);
		Sigma = inv(sigmav);
		CM((-1==yt)+1,(-1==ey)+1) = CM((-1==yt)+1,(-1==ey)+1)+1;
	end
end
function aResult = Test(mu,Sigma,test,L)
	[M,~]=size(test);
	CM = zeros(2,2);
	PositiveSample = sum(L==1);
	NegtiveSample = sum(L==-1);
	for t=1:M
		xt = test(t,:);
		yt = L(t);
		w = mvnrnd(mu,Sigma)';
		ey = sign(xt*w);
		CM((-1==yt)+1,(-1==ey)+1) = CM((-1==yt)+1,(-1==ey)+1)+1;
	end
	t=(CM(1,1)+CM(2,2))/(PositiveSample+NegtiveSample);
	CM(1,:) = CM(1,:)/PositiveSample;
	CM(2,:) = CM(2,:)/NegtiveSample;
	aResult = [reshape(CM,[1,4]),t];
end
function [mu,Sigma,CM]=Init(d)
	a = 0.05;CM=zeros(2,2);
	mu = zeros(d,1);Sigma = a*eye(d);
end
function SaveResult(dataset,CW_OPT)
	filename=strcat(dataset,'.mat');
	if(exist(filename,'file'))
		save(filename,'CW_OPT','-append');
	else
		save(filename,'CW_OPT');
	end
end
function xCW(dataset)
fprintf('*********CW Begin*********\n');
[label,train] = readdata(dataset,'train');
N=size(train,1);train = [train,ones(N,1)];
[N,d] = size(train);

[L,test] = readdata(dataset,'test');
M=size(test,1);test = [test,ones(M,1)];
[M,~] = size(test);

a = 0.05;

CW_history = [];

%问题：Sigma会出现负数，应该是数据问题。试试把数据缩放到[0,1] <已修正>
for yita = 0.7:0.05:0.95
	correct = 0;CM = zeros(2,2);
	fprintf('======yita: %f==========\n',yita);
	phi = norminv(yita,0,1);
	Sigma = a*eye(d);sigmav = inv(Sigma);
	mu = zeros(d,1);
	for t = 1:N
		xt = train(t,:);
		yt = label(t);
		w = mvnrnd(mu,Sigma)';
		ey = sign(xt*w);
		Mi = yt*(xt*mu);
		Vi = xt*Sigma*xt';
		gamma = ( -(1+2*phi*Mi) + sqrt( (1+2*phi*Mi)^2-8*phi*(Mi-phi*Vi) ) )/(4*phi*Vi);
		alpha = max(gamma,0);
		
		mu = mu+alpha*yt*Sigma*xt';
		sigmav = sigmav + 2*alpha*phi*diag(xt);
		Sigma = inv(sigmav);
		CM(int8(-1==yt)+1,int8(-1==ey)+1) = CM(int8(-1==yt)+1,int8(-1==ey)+1)+1;
		if(ey==yt)
			correct=correct+1;
		end
	end
	fprintf('Train. %d/%d\n',correct,N);
	fprintf(' tp:%.4f, fn:%.4f, fp:%.4f, tn:%.4f\n',CM(1,1)/sum(label==1),CM(1,2)/sum(label==1),CM(2,1)/sum(label==-1),CM(2,2)/sum(label==-1) );

	CM = zeros(2,2);
	PositiveSample = sum(L==1);
	NegtiveSample = sum(L==-1);
	for t=1:M
		xt = test(t,:);
		yt = L(t);
		w = mvnrnd(mu,Sigma)';
		ey = sign(xt*w);
		CM(int8(-1==yt)+1,int8(-1==ey)+1) = CM(int8(-1==yt)+1,int8(-1==ey)+1)+1;
	end
	CW_history = cat(1,CW_history,reshape(CM,[1,4]));
	fprintf('Test. %d/%d\n', CM(1,1)+CM(2,2),M);
	fprintf(' tp:%.4f, fn:%.4f, fp:%.4f, tn:%.4f\n',CM(1,1)/PositiveSample,CM(1,2)/PositiveSample,CM(2,1)/NegtiveSample,CM(2,2)/NegtiveSample);
end
filename=strcat(dataset,'.mat');
if(exist(filename,'file'))
	save(filename,'CW_history','-append');
else
	save(filename,'CW_history');
end
fprintf('*********CW End*********\n');
end