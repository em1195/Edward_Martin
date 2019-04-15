
results = [];
results = [results; create_net(14, X, t,'trainscg',0.3,'standard',500, 1e-10)];
results = [results; create_net(14, X, t,'trainscg',0.3,'standard',1000, 1e-10)];
results = [results; create_net(14, X, t,'trainscg',0.3,'standard',1500, 1e-10)];
results = [results; create_net(14, X, t,'trainscg',0.3,'standard',2000, 1e-10)];



more_cum = results;
hold off
plot(more_cum,'r')


