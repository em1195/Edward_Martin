results = [];
results = [results,
    create_net([9 9],raw,y_onehot),
    create_net([9 9],raw4,y_onehot),
    create_net([9 9],raw5,y_onehot),
    create_net([9 9],input_mean,y_onehot),
    create_net([9 9],mapped,y_onehot),
    create_net([9 9],meanmap,y_onehot),
    create_net([9 9],mapstded,y_onehot),
    ]