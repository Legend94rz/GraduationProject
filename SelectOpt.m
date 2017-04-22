function [valid,OptHistory] = SelectOpt(dataset,funcName,threshold)
%dataset='a7a';funcName='PA';threshold = 0.4;

fileName = strcat(dataset,'.mat');
varName = strcat(funcName,'_history');

if(exist(fileName,'file')==0)
	valid=0;OptHistory=[];return;
end
s = load(fileName,varName);
if(isfield(s,varName)==0)
	valid=0;OptHistory=[];return;
end
if(threshold<=0)
	valid=1;OptHistory=[]; return;
end
history=getfield(s,varName); %#ok<GFLD>

posNum = history(1,1)+history(1,3);
negNum = history(1,2)+history(1,4);
x = history(:,1)+history(:,4);
history(:,[1,3]) = history(:,[1,3])./posNum;
history(:,[2,4]) = history(:,[2,4])./negNum;

[N,~] = size(history);
m=0;j=0;
for i=1:N
	if(history(i,1)>=threshold && history(i,4)>=threshold && x(i)>m)
		m=x(i);
		j=i;
	end
end

if(j==0)
	j=randi([1,N]);
	fprintf('warning:random pick a result. dataset: %s, funcName: %s, j: %d\n',dataset,funcName,j);
end
OptHistory = [history(j,1:4),x(i)/(posNum+negNum)];
valid=1;
end
