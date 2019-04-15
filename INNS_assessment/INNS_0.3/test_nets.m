
hold off
final5 = [];
for i = 0:4
    results = [];
    for j = 4:2:14
        if i == 0
            results = [results; create_net(j, X,t)];
        elseif i == 1
            results = [results; create_net(j, X(2:4,:),t)];
        elseif i == 2
            results = [results; create_net(j, [X(1,:);X(3:4,:)],t)];
        elseif i == 3
            results = [results; create_net(j, [X(1:2,:);X(4,:)],t)];
        elseif i == 4
            results = [results; create_net(j, X(1:3,:),t)];
        end
        
        
    end
    final5 = [final5, results]
end




