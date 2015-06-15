function [tied_units] = initialize_tied_units (param, h_dim)
% generate list of tied units

n = 0;
tied_units = cell (param.tile_size^2,1);

for a = 1:param.tile_size
    for b = 1:param.tile_size
        n = n + 1;
        [tempx,tempy] = meshgrid ([a:param.tile_size:h_dim],[b:param.tile_size:h_dim]);
        temp = (tempy-1).*h_dim + tempx;
        temp = temp';
        tied_units{n} = reshape(temp, numel(temp), 1); 
    end
end

tied_units = repmat(tied_units, param.num_maps,1);

offset = [0:1:param.num_maps-1];
offset = cell2mat(arrayfun(@(x)x.*ones(1,param.tile_size^2)*h_dim^2,offset,'un',0));

for a = 1:length(offset)
    tied_units{a} = tied_units{a} + offset(a);
end

end
