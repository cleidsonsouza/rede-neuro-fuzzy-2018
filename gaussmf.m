function [ result ] = gaussmf(x, k)
	sig = k(1,1);
	c   = k(1,2);
	result = exp(-(x-c).^2/(2*sig^2));
end