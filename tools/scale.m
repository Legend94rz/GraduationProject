function ans = scale( v,a,b )
%SCALE �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
	l=min(v);
	r=max(v);
	ans = (b-a)/(r-l)*(v-l)+a;
end

