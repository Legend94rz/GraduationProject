%dataName=char({'svmguide1'});
%funcName=char({'PA','CW','SCW','LOL','Pegasos'});

dataName=char({'a7a','w7a','svmguide1','ijcnn1','cbcl'});
funcName=char({'PA','CW','SCW','LOL','Pegasos','BSGD'});
[dataLength,~] = size(dataName);
[funcLength,~]=size(funcName);
s = load('ALL.mat','ALL');
if isfield(s,'ALL')==0
	ALL = [];
	for i=1:dataLength
		dataset=strtrim(dataName(i,:));
		A = zeros(funcLength,4);
		for j=1:funcLength
			fun = str2func(strtrim(funcName(j,:)));
			optCM = reshape( fun(dataset) ,[1,4]);
			A(j,:) = optCM;
		end
		ALL = cat(3,ALL,A);
	end
else
	ALL=s.ALL;
end
figure;
y=1:4;
vcount = ceil(sqrt(dataLength));
hcount = ceil(dataLength/vcount);
for i=1:dataLength
	dataset=strtrim(dataName(i,:));
	subplot(vcount,hcount,i);
	A = ALL(:,:,i);
	b=bar(y,A','grouped');title(dataset);
	ax = b.Parent;
	ax.XTickLabel = {'tp','fp','fn','tn'};
	legend(funcName);
end
if isfield(s,'ALL')==0
	save ALL;
end