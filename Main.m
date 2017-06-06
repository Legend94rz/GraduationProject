function Main()
	keySet = {'svmguide1','a7a','w7a','cbcl','ijcnn1'};
	valSet = {{},...
			  {},...
			  {},...
			  {},...
			  {}};
	ReCals = containers.Map(keySet,valSet);

	dataNames=char({'svmguide1','a7a','w7a','cbcl','ijcnn1'});
	funcNames=char({'PA','CW','SCW','LOL','BSGD','Pegasos','Pegasos(single)'});
	[dataLength,~] = size(dataNames);
	[funcLength,~] = size(funcNames);
	ALL=[];
	for i=1:dataLength
		dataset=strtrim(dataNames(i,:));
		A = zeros(funcLength,5);
		for j=1:funcLength
			funcName = strtrim(funcNames(j,:));
			if(~strcmp(funcName,'Pegasos(single)'))
				[valid,~] = readOpt(dataset,funcName);
				if(valid==0 || (ReCals.isKey(dataset) && sum( ismember(ReCals(dataset),funcName)))>0 )
					fun = str2func(strtrim(funcNames(j,:)));
					fun(dataset);
				end
			else
				funcName='Pegasos_single';		%为了避免图例显示错误
			end
			[valid,optCM] = readOpt(dataset,funcName);
			A(j,:) = optCM;
		end
		ALL = cat(3,ALL,A);
	end
	y=1:5;
	vcount = ceil(sqrt(dataLength));
	hcount = ceil(dataLength/vcount);

	figure;
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

	figure;
	ratio=zeros(funcLength,1);
	for i=1:dataLength
		dataset=strtrim(dataNames(i,:));
		subplot(vcount,hcount,i);
		A = ALL(:,:,i);
		SortAndBarh(A(:,1),dataset,funcNames,0:0.2:1);
		ratio = ratio + A(:,1);
	end
	ratio = ratio/dataLength;
	figure;
	SortAndBarh(ratio,'平均真正例率',funcNames,0:0.2:1);
	
	figure;
	ratio = zeros(funcLength,1);
	for i=1:dataLength
		dataset=strtrim(dataNames(i,:));
		subplot(vcount,hcount,i);
		A = ALL(:,:,i);
		SortAndBarh(A(:,4),dataset,funcNames,0:0.2:1);
		ratio = ratio+A(:,4);
	end
	ratio = ratio/dataLength;
	figure;
	SortAndBarh(ratio,'平均真反例率',funcNames,0:0.2:1);

	figure;
	ratio = zeros(funcLength,1);
	for i=1:dataLength
		dataset=strtrim(dataNames(i,:));
		subplot(vcount,hcount,i);
		A = ALL(:,:,i);
		SortAndBarh(A(:,5),dataset,funcNames,0:0.2:1);
		ratio = ratio+A(:,5);
	end
	ratio = ratio/dataLength;
	figure;
	SortAndBarh(ratio,'平均精度',funcNames,0:0.2:1);
end
function SortAndBarh( InputData,Title,YTickLable,XTick )
	[Sorted,Index] = sort( InputData );
	b=barh(Sorted);title(Title);
	ax=b.Parent;
	ax.YTickLabel = YTickLable(Index,:);
	ax.XTick=XTick;
end

