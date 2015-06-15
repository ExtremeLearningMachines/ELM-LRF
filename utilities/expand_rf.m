function [W] = expand_rf (param, h_dim, tied_units, collapsed_rf)
% Duplicates unique units into sets of tied units (undos collapse rf)

W = zeros(h_dim^2*param.num_maps, size(collapsed_rf,2));

for a = 1:numel(tied_units)
    W(tied_units{a},:) = repmat(collapsed_rf(a,:),numel(tied_units{a}),1);
end

end
