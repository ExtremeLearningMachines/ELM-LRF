function [W_s] = shrink(index,W)
% Shrinks weight matrix down to contain only local non-zero entries
W_s = full(W)';
dim = [nnz(index(1,:)),size(index,1)];
W_s = reshape(W_s(index'),dim);
W_s = W_s';
end
