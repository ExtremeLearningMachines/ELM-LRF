function [a, grad] = squareroot_act(X)

a = sqrt(0.0001+X);

if nargout > 1
    grad = 0.5./a;  
end

end
