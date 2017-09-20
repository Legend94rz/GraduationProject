function [ y , x ] = readdata( whichset, usage )
%{
	if(strcmp( whichset,'svmguide1' ))
		[y,x] = libsvmread(strcat( 'data\svmguide1p.',usage ));
		y(y==0)=-1;
	else
		if(strcmp( whichset,'cbcl' ))
			load(strcat('data\cbcl.',usage),'-mat');
		else
			if(strcmp(whichset,'ijcnn'))
				[y,x] = libsvmread(strcat('data\ijcnn1p.',usage));
				return;
			else
				fprintf('warning : unreconized data set name: %s\n',whichset)
				[y,x] = libsvmread(strcat('data\',whichset,'.',usage));
			end
		end
	end
	A=[y,x];
	[M,N] = size(A);
	v = randperm(M);
	for i=1:M
		y(i) = A(v(i),1);
		x(i,:) = A(v(i),2:N);
	end
	fprintf('DataSet: %s, usage: %s, positive: %d, negtive:%d\n',whichset,usage,sum(y==1),sum(y==-1));
%}

	load( strcat('data\',whichset,'mat.',usage),'-mat');
	fprintf('DataSet: %s, usage: %s, positive: %d, negtive:%d\n',whichset,usage,sum(y==1),sum(y==-1));
	
end
