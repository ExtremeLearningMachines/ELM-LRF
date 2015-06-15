function [layer1, layer2] = two_layer_forwardprop(X, W, pool_layer, layer1_act, layer2_act)
layer1 = layer1_act(W*X);    
clear W X;    
l2_input = full((pool_layer)*double(layer1));    
clear pool_layer;    
layer2 = layer2_act(l2_input);

end
