function [ptest] = testing(net,X,y)
ptrain = [];
pvalidate = [];
ptest = [];
 for i = 1:1000
     init(net);
     [trained_net, TR] = train(net,X,y);
     ptrain = [ptrain; TR.best_perf];
     pvalidate = [pvalidate; TR.best_vperf];
     ptest = [ptest; TR.best_tperf];
 end
 
ptrain = mean(ptrain);
pvalidate = mean(pvalidate);
ptest = mean(ptest);
thing = [ptrain; pvalidate; ptest];
end

