function [ v ] = kernel( Xp, X )
%KERNEL �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
	sigma = 0.005;
	C = bsxfun( @minus,X,Xp );
	v = exp( -sum(C.^2,2)/(2*sigma^2) )';
end

