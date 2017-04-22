function [ valid , OptCM ] = loadCalcResult( dataset,algorithmName )
	if(exist(dataset,'file')==0)
		valid = 0;OptCM=[];
		return;
	end
	s=load(dataset,algorithmName);
	if(isfield(s,algorithmName))
		valid=1;OptCM=reshape(getfield(s,algorithmName),[1,4]); %#ok<GFLD>
	else
		valid=0;OptCM=[];
	end
end

