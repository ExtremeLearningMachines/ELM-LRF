function [pool_index] = initialize_pooling_indices_my1 (param, h_dim)

pool_index=zeros(param.num_maps*h_dim^2, param.num_maps*h_dim^2);
for k=1:param.num_maps;
    for j=1:h_dim;
        for i=1:h_dim;
            temp=zeros(h_dim, h_dim);
            temp(max(i-param.pooling_size, 1) :min(i+param.pooling_size, h_dim), max(1, j-param.pooling_size): min(h_dim, j+param.pooling_size))=1;
            temp=reshape(temp, 1, h_dim^2);
            pool_index(h_dim^2*(k-1)+h_dim*(j-1)+i, (k-1)*h_dim^2+1:k*h_dim^2)=temp;
        end
    end
end

end

