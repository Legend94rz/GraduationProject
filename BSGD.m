%dataset = 'a7a';
function [BSGD_OPTCM] = BSGD(dataset,Epsilon)
fprintf('*********BSGD Begin*********\n');
[label,train] = readdata(dataset,'train');
N=size(train,1);train = [train,ones(N,1)];
[N,d] = size(train);

[L,test] = readdata(dataset,'test');
M=size(test,1);test = [test,ones(M,1)];
[M,~] = size(test);

BSGD_OPTCM = zeros(2,2);
maxCorrect=0;
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
	if(CM(1,1)+CM(2,2)>maxCorrect && CM(2,2)<=Epsilon*NegtiveSample && CM(1,1) <=Epsilon*PositiveSample)
		maxCorrect = CM(1,1)+CM(2,2);
		BSGD_OPTCM = CM;
	end
end
fprintf('Test. %d/%d\n',CM(1,1)+CM(2,2),M);
fprintf(' tp:%.4f, fn:%.4f, fp:%.4f, tn:%.4f\n',CM(1,1)/PositiveSample,CM(1,2)/PositiveSample,CM(2,1)/NegtiveSample,CM(2,2)/NegtiveSample);
end
BSGD_OPTCM(1,:)=BSGD_OPTCM(1,:)./sum(L==1);
BSGD_OPTCM(2,:)=BSGD_OPTCM(2,:)./sum(L==-1);
if(exist(strcat(dataset,'.mat'),'file'))
	save(strcat(dataset,'.mat'),'BSGD_OPTCM','-append');
else
	save(strcat(dataset,'.mat'),'BSGD_OPTCM');
end
fprintf('*********BSGD End*********\n');
end