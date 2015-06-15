function [ W, rf_index, pool_index, h_dim, tied_units ] = gen_weights_my2( param )

% generate orthogonal random weights
% (1) generate a 40368X2048 matrix
% (2) shrink.m into 40368X32
% (3) collapse.m into 48X32
% (4) orthogonalize into 48X32
% (5) expand_rf.m into 40368X32
% (6) full_size.m into 40368X2048

% pooling: my pooling codes (initialize_pooling_indices_my2.m)


% initialize indices for rf windows, pooling and weights tying
[rf_index, h_dim, num_windows]=initialize_rf_indices(param);

pool_index=initialize_pooling_indices_my2(param, h_dim);

tied_units=initialize_tied_units(param, h_dim);

% To make the results reproducible, we use a random seed 0;
randn('state', 0);

W(1:param.num_maps*h_dim^2,:) = randn(param.num_maps*h_dim^2,param.input_ch*param.image_size^2);
W = W.*rf_index;
W = shrink(rf_index,W);

% combine tied units into a single average or summed rf
W=collapse_rf(W, tied_units, param);

% orthogonalize collapsed rfs looking at the same patch of the image
for b=1:param.tile_size^2;
    n=0;
    for c=0:param.num_maps-1;
        n=n+1;
        ortho_index(n)=b+c*(param.tile_size^2);
    end
    
    temp=W(ortho_index, :);
    [m, n]=size(temp);
    if m<=n;
        W(ortho_index, :)=(orth(temp'))';
    else
        W(ortho_index, :)=orth(temp);
    end
end
end

