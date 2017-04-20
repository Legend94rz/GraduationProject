function [ v ] = kernel( x, SV, sigma )
	C = bsxfun( @minus,SV,x );
	v = exp( -sum(C.^2,2)/(2*sigma^2) )';
end

