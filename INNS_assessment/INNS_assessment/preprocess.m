function [new_X] = preprocess(X)
X = bsxfun(@minus,X,mean(X));
new_X = zscore(X);




end

