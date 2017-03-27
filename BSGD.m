dataset='cbcl';
[label,train] = readdata(dataset,'train');
N=size(train,1);train = [train,ones(N,1)];
[N,d] = size(train);

[L,test] = readdata(dataset,'test');
M=size(test,1);test = [test,ones(M,1)];
[M,~] = size(test);
for lambda = [0.0001,0.001,0.01,0.1,1,10,100,1000]
B=500;
alpha=zeros(B,1);
I=zeros(B,1);
fprintf('lambda: %d\n',lambda);
%%init
for i=1:B
	p = randi([1,N]);
	I(i) = p;
	alpha(i) = 1/lambda*label(p);
end
w=zeros(B,1);
for it=1:2*N
	t = randi([1,N]);
	xt = train(t,:);yt = label(t);
	yita = 1/(lambda*t);
	alpha = (1-yita*lambda)*alpha;
	beta = 0;
	w = (1-yita*lambda)*w;
	Phix = kernel(xt,train(I,:));
	if(yt*Phix*w<1)
		beta = yita*yt;
		w = w + beta * Phix'; %todo kernelize
		%[~,i] = min(alpha.^2);
		i=randi([1,B]);
		w = w - alpha(i) * kernel(train(i,:),train(I,:) )';
		alpha(i) = beta;  I(i) = t;		
	end
end
correct=0;
CM=zeros(2,2);
for t = 1:M
	xt = test(t,:);yt=L(t);
	ey = sign( kernel(xt,train(I,:))*w );
	CM(int8(-1==yt)+1,int8(-1==ey)+1) = CM(int8(-1==yt)+1,int8(-1==ey)+1)+1;
	if(ey==yt)
		correct=correct+1;
	end
end
fprintf('Test. %d/%d\n',correct,M);
fprintf(' tp:%.4f, fn:%.4f, fp:%.4f, tn:%.4f\n',CM(1,1)/sum(L==1),CM(1,2)/sum(L==1),CM(2,1)/sum(L==-1),CM(2,2)/sum(L==-1) );
end