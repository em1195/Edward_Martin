function [total] = evaluate(y,ind,onehot)
count1 = 0;
count2 = 0;
for i = 1:length(ind)
    if (y(1,ind(i))==1)
        if (onehot(1,ind(i))==1) 
            count1 = count1 + 1;
        end
    end
    if (y(2,ind(i))==1)
        if (onehot(2,ind(i))==1) 
            count1 = count1 + 1;
        end
    end
end

total = (count1+count2) / length(ind);
end