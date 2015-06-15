function [rf_index, h_dim, num_windows] = initialize_rf_indices (param)

% Create index for overlaping receptive fields (1 input channel)
h_dim = length(1:param.step:param.image_size-param.window_size+1);

rf_index = zeros(h_dim^2,param.image_size^2);

temp = [ones(param.window_size),zeros(param.window_size,param.image_size);zeros(param.image_size,param.window_size+param.image_size)];

n = 0;

for a = 0:h_dim-1
    for b = 0:h_dim-1
         n = n + 1; 
         curr_rf = circshift(temp,[a*param.step,b*param.step]);
         rf_index(n,:) = reshape(curr_rf(1:param.image_size,1:param.image_size),param.image_size^2,1);
    end
end

rf_index = repmat (rf_index, param.num_maps, 1);
num_windows = h_dim^2 * param.num_maps;

rf_index = repmat(rf_index,1,param.input_ch);

rf_index = sparse(logical(rf_index));

end
