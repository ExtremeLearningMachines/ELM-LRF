function [fullW] = full_size(W,index)
% Expands weight matrix to full size (after using shrink) by padding with zeros
fullW = zeros(size(index'));
fullW(index') = W';
fullW = fullW';
end
