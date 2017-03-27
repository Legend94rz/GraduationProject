dataset='w7a';
[label,train] = readdata(dataset,'train');
N=size(train,1);train = [train,ones(N,1)];
[N,d] = size(train);

[L,test] = readdata(dataset,'test');
M=size(test,1);test = [test,ones(M,1)];
[M,~] = size(test);

eta = 0.95;
for c=-4:4
C=10^c;
correct = 0;CM = zeros(2,2);
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
	if(ey==yt)
		correct=correct+1;
	end
end
fprintf('Train. %d/%d\n',correct,N);
fprintf(' tp:%.4f, fn:%.4f, fp:%.4f, tn:%.4f\n',CM(1,1)/sum(label==1),CM(1,2)/sum(label==1),CM(2,1)/sum(label==-1),CM(2,2)/sum(label==-1) );
correct = 0;CM = zeros(2,2);
for t = 1:M
	xt = test(t,:);
	yt = L(t);
	w = mvnrnd(mu,Sigma)';
	ey = sign(xt*w);
	CM(int8(-1==yt)+1,int8(-1==ey)+1) = CM(int8(-1==yt)+1,int8(-1==ey)+1)+1;
	if(ey==yt)
		correct=correct+1;
	end
end
fprintf('Test. %d/%d\n',correct,M);
fprintf(' tp:%.4f, fn:%.4f, fp:%.4f, tn:%.4f\n',CM(1,1)/sum(L==1),CM(1,2)/sum(L==1),CM(2,1)/sum(L==-1),CM(2,2)/sum(L==-1) );
end