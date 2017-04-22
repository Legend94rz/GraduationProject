function LOL(dataset)
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