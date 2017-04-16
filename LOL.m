function [OptCM] = LOL(dataset)
fprintf('*********LOL Begin*********\n');
C=1;
[label,train] = readdata(dataset,'train');
k=60;
lambda=1;
N=size(train,1);train = [train,ones(N,1)];
[N,d] = size(train);
W = zeros(d,k+1);
count = zeros(1,k);
P = zeros(k,d);
OptCM = zeros(2,2);
maxCorrect=0;
for t=1:N
	xt = train(t,:);
	i=t;
	if(t<=k)
		P(t,:) = xt;
		count(t)=count(t)+1;
	else
		%todo :这会带来性能问题
		[~,i] = min( sum( bsxfun( @(A,B) (A-B).^2, P, xt ) ,2 ) );
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
[L,test]=readdata(dataset,'test');
%load('mydatafortest.mat');
M=size(test,1);test = [test,ones(M,1)];
[M,~] = size(test);
correct = 0;
CM = zeros(2,2);
for t = 1:M
	xt = test(t,:);
	Xt = zeros(k+1,d);
	Xt(1,:) = xt;
	[~,i] = min( sum( bsxfun( @(A,B) (A-B).^2, P, xt ) ,2 ) );
	Xt(i,:) = xt;
	result = Xt*W;
	ey = sign( sum( result(:) ) );
	yt=L(t);
	CM(int8(-1==yt)+1,int8(-1==ey)+1) = CM(int8(-1==yt)+1,int8(-1==ey)+1)+1;
	if(ey==yt)
		correct = correct+1;
	end
	if(correct>maxCorrect)
		maxCorrect = correct;
		OptCM = CM;
	end
end
fprintf('Test. %d/%d\n',correct,M);
fprintf(' tp:%.4f, fn:%.4f, fp:%.4f, tn:%.4f\n',CM(1,1)/sum(L==1),CM(1,2)/sum(L==1),CM(2,1)/sum(L==-1),CM(2,2)/sum(L==-1) );
OptCM(1,:)=OptCM(1,:)./sum(L==1);
OptCM(2,:)=OptCM(2,:)./sum(L==-1);
fprintf('*********LOL Begin*********\n');
end