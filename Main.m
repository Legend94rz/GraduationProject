Recalc=0;
dataNames=char({'svmguide1','a7a','w7a','cbcl','ijcnn1'});
funcNames=char({'PA','CW','SCW','LOL','BSGD','Pegasos'});
[dataLength,~] = size(dataNames);
[funcLength,~] = size(funcNames);
s = load('ALL.mat','ALL');
if isfield(s,'ALL')==0
	ALL = [];
	for i=1:dataLength
		dataset=strtrim(dataNames(i,:));
		A = zeros(funcLength,4);
		for j=1:funcLength
			funcName = strtrim(funcNames(j,:));
			[valid,optCM] = loadCalcResult(strcat(dataset,'.mat'),strcat(funcName,'_OPTCM'));
			if(valid==0 || Recalc~=0)
				fun = str2func(strtrim(funcNames(j,:)));
				optCM = reshape( fun(dataset,0.99) ,[1,4]);
			end
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
	dataset=strtrim(dataNames(i,:));
	subplot(vcount,hcount,i);
	A = ALL(:,:,i);
	b=bar(y,A','grouped');title(dataset);
	ax = b.Parent;
	ax.XTickLabel = {'tp','fp','fn','tn'};
	legend(funcNames);
end
if isfield(s,'ALL')==0
	save ALL;
end