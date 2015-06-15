function [collapsed_rf] = collapse_rf (W, tied_units, param) 
% "Collapses" tied units into a single averaged unit

collapsed_rf = zeros(length(tied_units), size(W,2));

for a = 1: numel(tied_units)
    collapsed_rf(a,:) = (sum(W(tied_units{a},:),1))/(length(tied_units{a} + 1e-10));
end

end

