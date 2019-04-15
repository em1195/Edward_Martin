data = csvread('dataR2.csv',1,0);
labels = {'age','bmi','glucose','insulin', 'homa','leptin','adiponectin','resistin','mcp1','class'};
X = data(:,1:9);
X = [X(:,1:3),X(:,8)];
Xpca = X;
y = data(:,10);
y_onehot = [zeros(2,116)];
y_onehot(1,1:52) = 1;
y_onehot(2,53:116) = 1;

c1 = X(1:52,:);
c2 = X(53:116,:);



X = bsxfun(@minus,X,mean(X))';
t = y_onehot;
