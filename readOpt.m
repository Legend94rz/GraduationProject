function [valid, aResult ] = readOpt( datasetName,algName )
	fileName = strcat(datasetName,'.mat');
	varName = strcat(algName,'_OPT');
	
	if(exist(fileName,'file')==0)
		valid=0;aResult=[];return;
	end
	s = load(fileName,varName);
	if(isfield(s,varName)==0)
		valid=0;aResult=[];return;
	end
	
	aResult=getfield(s,varName); %#ok<GFLD>
	valid=1;
end

