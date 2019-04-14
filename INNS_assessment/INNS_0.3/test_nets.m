results = [];
results = [results; create_net(2, X, t)];
results = [results; create_net(4, X, t)];
results = [results; create_net(6, X, t)];
results = [results; create_net(8, X, t)];
results = [results; create_net(10, X, t)];

hold off
plot(results)


