keySet = {'svmguide1','a7a','w7a','cbcl','ijcnn1'};
valSet = {{'PA','CW','SCW','LOL','BSGD','Pegasos'},	...
		  {'PA','CW','SCW','LOL','BSGD','Pegasos'},...
		  {'PA','CW','SCW','LOL','BSGD','Pegasos'},...
		  {'PA','CW','SCW','LOL','BSGD','Pegasos'},...
		  {'PA','CW','SCW','LOL','BSGD','Pegasos'}};
ReCals = containers.Map(keySet,valSet);

dataNames=char({'svmguide1','a7a','w7a','cbcl','ijcnn1'});
funcNames=char({'PA','CW','SCW','LOL','BSGD','Pegasos'});
[dataLength,~] = size(dataNames);
[funcLength,~] = size(funcNames);
ALL=[];
for i=1:dataLength
	dataset=strtrim(dataNames(i,:));
	A = zeros(funcLength,4);
	for j=1:funcLength
		funcName = strtrim(funcNames(j,:));
		[valid,optCM] = loadCalcResult(strcat(dataset,'.mat'),strcat(funcName,'_OPTCM'));
		if(valid==0 || (ReCals.isKey(dataset) && sum( ismember(ReCals(dataset),funcName)))>0 )
			fun = str2func(strtrim(funcNames(j,:)));
			optCM = reshape( fun(dataset,0.98) ,[1,4]);
		end
		A(j,:) = optCM;
	end
	ALL = cat(3,ALL,A);
end
figure;
y=1:4;
vcount = ceil(sqrt(dataLength));
hcount = ceil(dataLength/vcount);
for i=1:dataLength
	dataset=strtrim(dataNames(i,:));
	subplot(vcount,hcount,i);
	A = ALL(:,:,i);
	b=bar(y,A','grouped');title(dataset);
	ax = b.Parent;
	ax.XTickLabel = {'tp','fp','fn','tn'};
	legend(funcNames);
end
save('ALL.mat','ALL');