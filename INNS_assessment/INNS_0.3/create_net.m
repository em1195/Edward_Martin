function [performance] = create_net(arch, x,t,algo,reg,norm, epochs, min_grad)


net = patternnet(arch);
init(net);
net.trainFcn = algo;
net.performParam.regularization = reg;
net.performParam.normalization = norm;
net.trainParam.epochs = epochs;
net.trainParam.min_grad = min_grad;
net
net.trainParam
%performance = 1 - test_net(net,x,t)

end