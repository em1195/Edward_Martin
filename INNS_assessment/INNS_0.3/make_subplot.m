
hold on
count = 0
for i = 1:9
    
        count = count + 1
        subplot(9,1,count)
        
        
        hold on
        plot(X(i,1:52),'bo')
        plot(X(j,53:116),'ro')
    end
    

hold off