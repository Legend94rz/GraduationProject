function SCW(dataset)
	fprintf('*********SCW Begin*********\n');
	load(strcat('data\',dataset,'.mat'));
	N=size(train,1);train = [train,ones(N,1)];
	[N,d] = size(train);
	
	[L,test] = readdata(dataset,'test');
	M=size(test,1);test = [test,ones(M,1)];
	[M,~] = size(test);
	
	SCW_history=[];mu_history=[];sigma_history=[];
	nperpart =N/fold;eta = 0.95;
	for c=-4:4
		C=10^c;
		fprintf('======C: %d==========\n',C);
		rule = (1:fold)';
		result=zeros(1,5);
		for TestRound=1:fold
			[mu,sigma,CM] = Init(d);
			for partion = 1:fold-1
				iv = (rule(partion)-1)*nperpart+1 : rule(partion)*nperpart;
				[mu,sigma]=Train(eta,C,mu,sigma,CM,train( iv ,: ),label(iv) );
			end
			iv=(rule(fold)-1)*nperpart+1:rule(fold)*nperpart;
			aResult = Test( mu,sigma,train( iv ,:),label(iv) );
			result = (aResult+(TestRound-1)*result)/TestRound;
			fprintf(' tp:%.4f, fn:%.4f, fp:%.4f, tn:%.4f, P:%.4f\n',aResult(1),aResult(2),aResult(3),aResult(4),aResult(5) );
			rule=circshift(rule,1);
		end
		fprintf('average: tp:%.4f, fn:%.4f, fp:%.4f, tn:%.4f, P:%.4f\n',result(1),result(2),result(3),result(4),result(5) );
		SCW_history = cat(1,SCW_history,result);
		mu_history = cat(1,mu_history,mu');
		sigma_history = cat(3,sigma_history,sigma);
	end
	fprintf('summery: \n');
	disp(SCW_history);
	[~,~,location] = SelectOpt(SCW_history);
	mu = mu_history(location,:)';
	sigma=sigma_history(:,:,location);
	finalResult = Test(mu,sigma,test,L);
	fprintf('final, optIndex: %d\n',location);
	disp(finalResult);
	SaveResult(dataset, finalResult);
	fprintf('*********SCW End*********\n');
end
function aResult = Test(mu,Sigma,test,L)
	[M,~]=size(test);
	CM = zeros(2,2);
	PositiveSample = sum(L==1);
	NegtiveSample = sum(L==-1);
	for t = 1:M
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
function [mu,Sigma]=Train(eta,C,mu,Sigma,CM,train,label)
	phi = norminv(eta,0,1);[N,~]=size(train);
	for t=1:N
		xt = train(t,:);
		yt = label(t);
		w = mvnrnd(mu,Sigma)';
		ey = sign(xt*w);
		loss = max(0,phi*sqrt(xt*Sigma*xt')-yt*xt*mu);
		if(loss>0)
			vt=xt*Sigma*xt';mt=yt*(xt*mu);zeta = 1+phi^2;
			psa = 1+phi^2/2;
			alpha=min(C,max(0, (-mt*psa + sqrt(mt^2*phi^4/4+vt*phi^2*zeta)) / (vt*zeta)));
			ut = 1/4*(-alpha*vt*phi+sqrt(alpha^2*vt^2*phi^2+4*vt))^2;
			beta=alpha*phi/(sqrt(ut)+vt*alpha*phi);
			
			mu = mu+alpha*yt*Sigma*xt';
			Sigma = Sigma-beta*Sigma*(xt*xt')*Sigma;
		end
		CM((-1==yt)+1,(-1==ey)+1) = CM((-1==yt)+1,(-1==ey)+1)+1;
	end
end
function [mu,Sigma,CM]=Init(d)
	CM = zeros(2,2);
	mu = zeros(d,1);
	Sigma=eye(d);
end
function SaveResult(dataset,SCW_OPT)
	filename=strcat(dataset,'.mat');
	if(exist(filename,'file'))
		save(filename,'SCW_OPT','-append');
	else
		save(filename,'SCW_OPT');
	end
end
function xSCW(dataset)
fprintf('*********SCW Begin*********\n');
[label,train] = readdata(dataset,'train');
N=size(train,1);train = [train,ones(N,1)];
[N,d] = size(train);

[L,test] = readdata(dataset,'test');
M=size(test,1);test = [test,ones(M,1)];
[M,~] = size(test);

eta = 0.95;

SCW_history=[];

for c=-4:4
C=10^c;
CM = zeros(2,2);
fprintf('======C: %f==========\n',C);
mu = zeros(d,1);
Sigma=eye(d);
phi = norminv(eta,0,1);
for t=1:N
	xt = train(t,:);
	yt = label(t);
	w = mvnrnd(mu,Sigma)';
	ey = sign(xt*w);
	loss = max(0,phi*sqrt(xt*Sigma*xt')-yt*xt*mu);
	if(loss>0)
		vt=xt*Sigma*xt';mt=yt*(xt*mu);zeta = 1+phi^2;
		psa = 1+phi^2/2;
		alpha=min(C,max(0, (-mt*psa + sqrt(mt^2*phi^4/4+vt*phi^2*zeta)) / (vt*zeta)));
		ut = 1/4*(-alpha*vt*phi+sqrt(alpha^2*vt^2*phi^2+4*vt))^2;
		beta=alpha*phi/(sqrt(ut)+vt*alpha*phi);
		
		mu = mu+alpha*yt*Sigma*xt';
		Sigma = Sigma-beta*Sigma*(xt*xt')*Sigma;
	end
	CM(int8(-1==yt)+1,int8(-1==ey)+1) = CM(int8(-1==yt)+1,int8(-1==ey)+1)+1;
end
fprintf('Train. %d/%d\n',CM(1,1)+CM(2,2),N);
fprintf(' tp:%.4f, fn:%.4f, fp:%.4f, tn:%.4f\n',CM(1,1)/sum(label==1),CM(1,2)/sum(label==1),CM(2,1)/sum(label==-1),CM(2,2)/sum(label==-1) );
CM = zeros(2,2);
PositiveSample = sum(L==1);
NegtiveSample = sum(L==-1);
for t = 1:M
	xt = test(t,:);
	yt = L(t);
	w = mvnrnd(mu,Sigma)';
	ey = sign(xt*w);
	CM(int8(-1==yt)+1,int8(-1==ey)+1) = CM(int8(-1==yt)+1,int8(-1==ey)+1)+1;
end
SCW_history = cat(1,SCW_history,reshape(CM,[1,4]));
fprintf('Test. %d/%d\n',CM(1,1)+CM(2,2),M);
fprintf(' tp:%.4f, fn:%.4f, fp:%.4f, tn:%.4f\n',CM(1,1)/PositiveSample,CM(1,2)/PositiveSample,CM(2,1)/NegtiveSample,CM(2,2)/NegtiveSample);
end
filename=strcat(dataset,'.mat');
if(exist(filename,'file'))
	save(filename,'SCW_history','-append');
else
	save(filename,'SCW_history');
end
fprintf('*********SCW End*********\n');
end