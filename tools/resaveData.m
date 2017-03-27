dataset='w7a';
usage='train';
[y,x] = readdata(dataset,usage);
save(strcat('data\',dataset,'mat.',usage),'x','y');