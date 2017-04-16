
dataName=char({'a7a','w7a','svmguide1','ijcnn1','cbcl'});
funcName=char({'PA','CW','SCW','LOL','Pegasos','BSGD'});
%dataName=char({'svmguide1'});
%funcName=char({'PA','CW','SCW','LOL','Pegasos'});

[dataLength,~] = size(dataName);
[funcLength,~]=size(funcName);
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

figure;
y=1:4;
vcount = ceil(sqrt(dataLength));
hcount = ceil(dataLength/vcount);
for i=1:dataLength
	dataset=strtrim(dataName(i,:));
	subplot(vcount,hcount,i);
	A = ALL(:,:,i);
	b=barh(y,A','grouped');title(dataset);
	ax = b.Parent;
	ax.YTickLabel = {'tp','fp','fn','tn'};
	legend(funcName);
end
save ALL;