function [coeff_out,score_out,latent_out,tsquared_out,explained_out,c1_out,c2_out] = perform_pca(X)

[coeff,score,latent,tsquared,explained] = pca(X);
c1 = X(1:52,:);
c2 = X(53:116,:);

coeff_out = coeff;
score_out = score;
latent_out = latent;
tsquared_out = tsquared;
explained_out = explained;
c1_out = c1;
c2_out = c2;

end

