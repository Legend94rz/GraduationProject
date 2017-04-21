function [LOL_OPTCM] = LOL(dataset,Epsilon)
fprintf('*********LOL Begin*********\n');

[label,train] = readdata(dataset,'train');
N=size(train,1);train = [train,ones(N,1)];
[N,d] = size(train);

[L,test]=readdata(dataset,'test');
M=size(test,1);test = [test,ones(M,1)];
[M,~] = size(test);

C=1;k=60;lambda=1;

W = zeros(d,k+1);
count = zeros(1,k);
P = zeros(k,d);
LOL_OPTCM = zeros(2,2);
maxCorrect=0;
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
	Xt(1,:) = xt;
	Xt(i,:) = xt;
	result = Xt*W;
	tmp = Xt.^2;
	yt = sign( sum(result(:)) );
	y = label(t);
	loss = max(0,1-y*sum(result(:)) );
	yita = min(C,loss/sum(tmp(:)));
	W = W + yita * y * Xt';
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
	ey = sign( sum( result(:) ) );
	yt=L(t);
	CM(int8(-1==yt)+1,int8(-1==ey)+1) = CM(int8(-1==yt)+1,int8(-1==ey)+1)+1;
	if(CM(1,1)+CM(2,2)>maxCorrect && CM(2,2)<=Epsilon*NegtiveSample && CM(1,1) <=Epsilon*PositiveSample)
		maxCorrect = CM(1,1)+CM(2,2);
		LOL_OPTCM = CM;
	end
end
fprintf('Test. %d/%d\n', CM(1,1)+CM(2,2),M);
fprintf(' tp:%.4f, fn:%.4f, fp:%.4f, tn:%.4f\n',CM(1,1)/PositiveSample,CM(1,2)/PositiveSample,CM(2,1)/NegtiveSample,CM(2,2)/NegtiveSample);
LOL_OPTCM(1,:)=LOL_OPTCM(1,:)./sum(L==1);
LOL_OPTCM(2,:)=LOL_OPTCM(2,:)./sum(L==-1);
if(exist(strcat(dataset,'.mat'),'file'))
	save(strcat(dataset,'.mat'),'LOL_OPTCM','-append');
else
	save(strcat(dataset,'.mat'),'LOL_OPTCM');
end
fprintf('*********LOL End*********\n');
end