function [performance] = create_net(arch, x,t)


net = patternnet(arch);
init(net);
net.trainFcn = 'trainscg';
net.performParam.regularization = 0.3;
net.performParam.normalization = 'standard';
net.trainParam.epochs = 1000;
performance = 1 - test_net(net,x,t);

end