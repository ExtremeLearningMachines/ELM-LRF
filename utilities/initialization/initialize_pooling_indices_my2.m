function [pool_index] = initialize_pooling_indices_my2 (param, h_dim)

num_pool = 0;
pool_index = [];
spatial_pool_index = eye(h_dim^2);
poolstep = 1;

% Spatial pooling
if param.pooling_size ~= 0

    p_dim = floor(h_dim/poolstep);

    spatial_pool_index = zeros(p_dim^2,h_dim^2);

    temp = [ones(param.pooling_size*2+1),zeros(param.pooling_size*2+1,h_dim);zeros(h_dim,param.pooling_size*2+1+h_dim)];
    n = 0;

    for a = 0:p_dim-1
        for b = 0:p_dim-1
            n = n + 1;
            curr_pool = circshift(temp,[b*poolstep,a*poolstep]);
            % changed
            spatial_pool_index(n,:) = reshape(curr_pool(param.pooling_size+1:h_dim+param.pooling_size,param.pooling_size+1:h_dim+param.pooling_size),h_dim^2,1);
        end
    end
    
    expanded_pool_index = zeros(p_dim^2*param.num_maps,h_dim^2*param.num_maps);
    % Replicate for num maps
    for a = 0:param.num_maps-1
        expanded_pool_index(a*p_dim^2 + 1:(a+1)*p_dim^2, a*h_dim^2+1 : (a+1)*h_dim^2 ) = spatial_pool_index;
    end
    
    pool_index = expanded_pool_index;
    num_pool = p_dim^2*param.num_maps;        
end

pool_index = sparse(logical(pool_index));

end

