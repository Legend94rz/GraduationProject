function [ v ] = kernel( Xp, X )
%KERNEL 此处显示有关此函数的摘要
%   此处显示详细说明
	sigma = 0.005;
	C = bsxfun( @minus,X,Xp );
	v = exp( -sum(C.^2,2)/(2*sigma^2) )';
end

