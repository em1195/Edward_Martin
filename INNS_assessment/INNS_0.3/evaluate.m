function [total] = evaluate(y)
count1 = 0;
count2 = 0;
for i = 1:52
    if (y(1,i)==1)
        count1 = count1 + 1;
    end
end
for i = 53:116
    if (y(2,i)==1)
        count2 = count2 + 1;
    end
end
total = (count1+count2) / 116;
end

