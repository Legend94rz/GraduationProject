function [valid,OptHistory,location] = SelectOpt(inHistory)

if (isempty(inHistory))
	fileName = strcat(dataset,'.mat');
	varName = strcat(funcName,'_history');
	
	if(exist(fileName,'file')==0)
		valid=0;OptHistory=[];location=0;return;
	end
	s = load(fileName,varName);
	if(isfield(s,varName)==0)
		valid=0;OptHistory=[];location=0;return;
	end
	
	history=getfield(s,varName); %#ok<GFLD>
else
	history=inHistory;
end

threshold=0.3;
[N,~] = size(history);
m=0;j=0;
for i=1:N
	if(history(i,1)>=threshold && history(i,4)>=threshold && history(i,5)>m)
		m=history(i,5);
		j=i;
	end
end

if(j==0)
	j=randi([1,N]);
	fprintf('warning:random pick a result. j: %d\n',j);
end
OptHistory = history(j,:);
valid=1;
location = j;
end
