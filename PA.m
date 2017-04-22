function PA(dataset)
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