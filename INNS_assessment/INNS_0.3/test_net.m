function [final] = test_net(net,x,t)
test_num = 100;
count = 0;
count_list = []
for i = 1:test_num
    init(net);
    [trnet, tr] = train(net,x,t);  
    results = round(trnet(x));
    new_results = evaluate(results,tr.testInd,t);
    count_list = [count_list; 1-new_results];
    count = count + new_results;
    results = [];
    new_results = [];
end
final = count/test_num;
end