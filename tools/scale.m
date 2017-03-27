function ans = scale( v,a,b )
%SCALE 此处显示有关此函数的摘要
%   此处显示详细说明
	l=min(v);
	r=max(v);
	ans = (b-a)/(r-l)*(v-l)+a;
end

