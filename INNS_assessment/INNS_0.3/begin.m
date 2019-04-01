data = csvread('dataR2.csv',1,0);
labels = {'age','bmi','glucose','homa','leptin','adiponectin','resistin','mcp','class'};
X = data(:,1:9);
y = data(:,10);
y_onehot = [zeros(2,116)];
y_onehot(1,1:52) = 1;
y_onehot(2,53:116) = 1;

c1 = X(1:52,:);
c2 = X(53:116,:);
raw = X';
raw4 = [raw(1:3,:);raw(5:9,:)];
raw5 = [raw(1:4,:);raw(6:9,:)];

input_mean = bsxfun(@minus,X,mean(X))';
input_mean = [input_mean(1:3,:);input_mean(5:9,:)];
mapped = mapminmax(raw5);
meanmap = mapminmax(input_mean);
mapstded = mapstd(raw4);


input_target = y'-1;
