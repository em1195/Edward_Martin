data = csvread('dataR2.csv',1,0);
labels = ["age",'bmi','glucose','insulin','homa','leptin','adiponectin','resistin','mcp','class'];
X = data(:,1:9);
y = data(:,10);
[coeff,score,latent,tsquared,explained,c1,c2] = perform_pca(X);