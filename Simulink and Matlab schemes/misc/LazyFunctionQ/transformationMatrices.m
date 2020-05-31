function A=transformationMatrices(q)
d=[0.333; 0; 0.316; 0; 0.384; 0; 0;0.107];
theta=[q;0];
a=[0; 0; 0; 0.0825; -0.0825; 0; 0.088; 0];
alfa=[0;-pi/2;pi/2;pi/2;-pi/2;pi/2;pi/2;0];

b = zeros(4,4,8);
for i=1:8
    b(:,:,i)=denavitHartenberg(d(i),theta(i),a(i),alfa(i));
end

A = b