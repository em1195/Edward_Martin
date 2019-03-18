function [outputArg1,outputArg2] = show_pc(pcX,pcY,c1,c2)
hold off
plot(c1(:,pcX),c1(:,pcY),'bo')
hold on
plot(c2(:,pcX),c2(:,pcY),'ro')
title('top 3 PCs')
xlabel('PCX')
ylabel('PCY')
outputArg1 = pcX;
outputArg2 = pcY;
end

