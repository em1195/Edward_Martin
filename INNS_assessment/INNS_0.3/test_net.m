function [final] = test_net(net,x,t)
test_num = 1000;
count = 0;
for i = 1:test_num
    init(net);
    [trnet, tr] = train(net,x,t);  
    results = round(trnet(x));
    count = count + evaluate(results,tr.testInd,t);
end
final = count/test_num;
end