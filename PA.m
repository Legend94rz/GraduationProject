function PA(dataset)
	fprintf('*********PA Begin*********\n');
	[L,test] = readdata(dataset,'test');
	M=size(test,1);test = [test,ones(M,1)];
	[M,~] = size(test);
	
	load(strcat('data\',dataset,'.mat'));
	N=size(train,1);train = [train,ones(N,1)];
	[N,d] = size(train);
	PA_history = [];w_history=[];nperpart =N/fold;
	for C=[0.001,0.01,0.1,1,10]
		fprintf('======C: %d==========\n',C);
		rule = (1:fold)';
		result=zeros(1,5);
		for TestRound=1:fold
			[w,CM] = Init(d);
			for partion = 1:fold-1
				iv = (rule(partion)-1)*nperpart+1 : rule(partion)*nperpart;
				w=Train(C,w,CM,train( iv ,: ),label(iv) );
			end
			iv=(rule(fold)-1)*nperpart+1:rule(fold)*nperpart;
			aResult = Test( w,train( iv ,:),label(iv) );
			result = (aResult+(TestRound-1)*result)/TestRound;
			fprintf(' tp:%.4f, fn:%.4f, fp:%.4f, tn:%.4f, P:%.4f\n',aResult(1),aResult(2),aResult(3),aResult(4),aResult(5) );
			rule=circshift(rule,1);
		end
		fprintf('average: tp:%.4f, fn:%.4f, fp:%.4f, tn:%.4f, P:%.4f\n',result(1),result(2),result(3),result(4),result(5) );
		PA_history = cat(1,PA_history,result);
		w_history = cat(1,w_history,w');
	end
	fprintf('summery on %s: \n',dataset);
	disp(PA_history);
	[~,~,location] = SelectOpt(PA_history);
	w = w_history(location,:)';
	finalResult = Test(w,test,L);
	fprintf('final, optIndex: %d\n',location);
	disp(finalResult);
	SaveResult(dataset, finalResult);
	fprintf('*********PA End*********\n');
end
function [w,CM] = Init(d)
	w = zeros(d,1);
	CM=zeros(2,2);
end
function  w = Train(C,w,CM,train,label)
	[N,~] = size(train);
	for t = 1:N
		xt = train(t,:);
		ey = sign(xt*w);
		yt = label(t);
		CM((-1==yt)+1,(-1==ey)+1) = CM((-1==yt)+1,(-1==ey)+1)+1;
		loss = max(0,1-yt*(xt*w));
		lambda = min(C,loss/sum(xt.^2));
		w = w+lambda*yt*xt';
	end
end
function aResult = Test(w,test,L)
	[M,~] = size(test);
	CM = zeros(2,2);
	PositiveSample = sum(L==1);
	NegtiveSample = sum(L==-1);
	for t=1:M
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
function SaveResult(dataset,PA_OPT)
	filename=strcat(dataset,'.mat');
	if(exist(filename,'file'))
		save(filename,'PA_OPT','-append');
	else
		save(filename,'PA_OPT');
	end
end

function xPA(dataset)
fprintf('*********PA Begin*********\n');
[label,train] = readdata(dataset,'train');
N=size(train,1);train = [train,ones(N,1)];
[N,d] = size(train);

[L,test] = readdata(dataset,'test');
M=size(test,1);test = [test,ones(M,1)];
[M,~] = size(test);

PA_history = [];

for C=[0.0001,0.001,0.01,0.1,1,10,100,1000,10000]
fprintf('======C: %d==========\n',C);
%C = 0.001;

w = zeros(d,1);
PE = zeros(N,1);
correct=0;
CM = zeros(2,2);%[tp,fn;fp,tn]
for t = 1:N
	xt = train(t,:);
	ey = sign(xt*w);
	PE(t) = ey;
	yt = label(t);
	CM(int8(-1==yt)+1,int8(-1==ey)+1) = CM(int8(-1==yt)+1,int8(-1==ey)+1)+1;
	if(ey==yt)
		correct=correct+1;
	end
	loss = max(0,1-yt*(xt*w));
	lambda = min(C,loss/sum(xt.^2));
	w = w+lambda*yt*xt';
end
fprintf('Train. %d/%d\n',correct,N);
fprintf(' tp:%.4f, fn:%.4f, fp:%.4f, tn:%.4f\n',CM(1,1)/sum(label==1),CM(1,2)/sum(label==1),   CM(2,1)/sum(label==-1),CM(2,2)/sum(label==-1) );
CM = zeros(2,2);
E = zeros(size(L));
PositiveSample = sum(L==1);
NegtiveSample = sum(L==-1);
for t=1:M
	xt = test(t,:);
	ey = sign(xt*w);
	E(t) = ey;
	yt = L(t);
	CM(int8(-1==yt)+1,int8(-1==ey)+1) = CM(int8(-1==yt)+1,int8(-1==ey)+1)+1;
end
PA_history = cat(1,PA_history,reshape(CM,[1,4]));
fprintf('Test. %d/%d\n', CM(1,1)+CM(2,2),M);
fprintf(' tp:%.4f, fn:%.4f, fp:%.4f, tn:%.4f\n',CM(1,1)/PositiveSample,CM(1,2)/PositiveSample,CM(2,1)/NegtiveSample,CM(2,2)/NegtiveSample);
end
filename=strcat(dataset,'.mat');
if(exist(filename,'file'))
	save(filename,'PA_history','-append');
else
	save(filename,'PA_history');
end
fprintf('*********PA End*********\n');
end