data = csvread('dataR2.csv',1,0);
labels = {'age','bmi','glucose','homa','leptin','adiponectin','resistin','mcp','class'};
X = data(:,1:9);
y = data(:,10);
c1 = X(1:52,:);
c2 = X(53:116,:);
input_data = mapminmax([X(:,1:3),X(:,5:9)]');
input_target = y';
