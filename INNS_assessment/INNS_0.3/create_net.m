function [performance] = create_net(arch, x,t)


net = patternnet(arch);
init(net);
performance = 1 - test_net(net,x,t)

end