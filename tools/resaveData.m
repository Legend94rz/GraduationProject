fold=5;
dataset='w7a';
[l,instance]=readdata(dataset,'train');
train=[];
label=[];
for index=1:fold
	pos = sum(l==1);
	neg = sum(l==-1);
	posSet=instance(l==1,:);
	negSet=instance(l==-1,:);
	posPerFold = floor( pos/fold );
	negPerFold = floor(neg/fold);
	p = 1 + posPerFold*(index-1) : posPerFold*index;
	n = 1 + negPerFold*(index-1) : negPerFold*index;
	A = [posSet(p,:),ones(posPerFold,1);negSet(n,:),-ones(negPerFold,1)];
	
	[M,N] = size(A);
	r = randperm(M);
	tx = A(r,1:N-1);
	ty = A(r,N);
	train=cat(1,train,tx);
	label=cat(1,label,ty);
end
save( strcat('data\',dataset,'.mat'),'train','label','fold');