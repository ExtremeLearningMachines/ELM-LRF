function [W] = initialize_W_my1(param, h_dim, rf_index)

W=randn(param.num_maps, input_ch*param.window_size^2);

end
