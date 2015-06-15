function [a, grad] = square_act(X)

a = X.^2;

if nargout > 1
    grad = 2*X;
end

end
