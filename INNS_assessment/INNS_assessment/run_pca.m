data = csvread('dataR2.csv',1,0);
labels = ["age",'bmi','glucose','insulin','homa','leptin','adiponectin','resistin','mcp','class'];
X = data(:,1:9);
y = data(:,10);

new_X = preprocess(X);
c1 = new_X(1:52,:);
c2 = new_X(53:116,:);
[coeff,score,latent,tsquared,explained,c1_score,c2_score] = perform_pca(new_X);
data_meanvar;