dataset='w7a';
[label,train] = readdata(dataset,'train');
N=size(train,1);train = [train,ones(N,1)];
[N,d] = size(train);

[L,test] = readdata(dataset,'test');
M=size(test,1);test = [test,ones(M,1)];
[M,~] = size(test);

Center = zeros(2,d+1);
Count = zeros(2);
%correct = 0;
for i=1:N
	xt = train(i,:);
	yt = label(i);
	vec = [xt,yt];
	p=1;
	if(yt==-1)
		p=2;
	end
	if(Count(p)==0)
		Center(p,:)=vec;
		Count(p)=1;
	end
	Center(p,:) = Center(p,:) + 1/Count(p) * (vec-Center(p,:));
	Count(p) = Count(p)+1;
end
%fprintf('Train. %d/%d\n',correct,N);
%fprintf(' tp:%.4f, fn:%.4f, fp:%.4f, tn:%.4f\n',CM(1,1)/sum(label==1),CM(1,2)/sum(label==1),CM(2,1)/sum(label==-1),CM(2,2)/sum(label==-1) );

CM = zeros(2,2);
correct = 0;
for i=1:M
	xt = test(i,:);
	yt = L(i);
	vec = [xt,0];
	t=sum( (vec-Center(1,:)).^2 );
	if(t>sum( (vec-Center(2,:)).^2 ))
		p=2;
	else
		p=1;
	end
	ey = sign( Center(p,d+1) );
	
	CM(int8(-1==yt)+1,int8(-1==ey)+1) = CM(int8(-1==yt)+1,int8(-1==ey)+1)+1;
	if(ey==yt)
		correct=correct+1;
	end
end
fprintf('Test. %d/%d\n',correct,M);
fprintf(' tp:%.4f, fn:%.4f, fp:%.4f, tn:%.4f\n',CM(1,1)/sum(L==1),CM(1,2)/sum(L==1),CM(2,1)/sum(L==-1),CM(2,2)/sum(L==-1) );
