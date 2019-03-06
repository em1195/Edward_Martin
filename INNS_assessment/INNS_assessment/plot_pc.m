function [output2,output1] = plot_pc(c1,c2,pc1,pc2,pc3)

hold off
scatter3(c1(:,pc1),c1(:,pc2),c1(:,pc3))
hold on
scatter3(c2(:,pc1),c2(:,pc2),c2(:,pc3))

xlabel(pc1)
ylabel(pc2)
zlabel(pc3)

output1 = pc1;
output2 = pc2;
end

