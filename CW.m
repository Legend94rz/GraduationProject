function CW(dataset)
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