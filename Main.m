keySet = {'svmguide1','a7a','w7a','cbcl','ijcnn1'};
valSet = {{},...
		  {},...
		  {},...
		  {},...
		  {}};
ReCals = containers.Map(keySet,valSet);

dataNames=char({'svmguide1','a7a','w7a','cbcl','ijcnn1'});
funcNames=char({'PA','CW','SCW','LOL','BSGD','Pegasos'});
[dataLength,~] = size(dataNames);
[funcLength,~] = size(funcNames);
ALL=[];
for i=1:dataLength
	dataset=strtrim(dataNames(i,:));
	A = zeros(funcLength,5);
	for j=1:funcLength
		funcName = strtrim(funcNames(j,:));
		[valid,~] = SelectOpt(dataset,funcName,-1);
		if(valid==0 || (ReCals.isKey(dataset) && sum( ismember(ReCals(dataset),funcName)))>0 )
			fun = str2func(strtrim(funcNames(j,:)));
			fun(dataset);
		end
		[valid,optCM] = SelectOpt(dataset,funcName,0.3);
		A(j,:) = optCM;
	end
	ALL = cat(3,ALL,A);
end
figure;
y=1:5;
vcount = ceil(sqrt(dataLength));
hcount = ceil(dataLength/vcount);
for i=1:dataLength
	dataset=strtrim(dataNames(i,:));
	subplot(vcount,hcount,i);
	A = ALL(:,:,i);
	b=bar(y,A','grouped');title(dataset);
	ax = b.Parent;
	ax.XTickLabel = {'tp','fp','fn','tn','P'};
	ax.YTick=0:0.2:1;
end
legend(funcNames);
save('ALL.mat','ALL');