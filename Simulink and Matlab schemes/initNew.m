%RV variant

%PD regulator Values
Kp = 10;
Kd = 0.1;
%load("misc\LazyFunctionQ\recorded.mat");
load("misc\LazyFunctionQ\recorded2.mat");
pts=single(pts);
if size(pts,1)==1
    pts(2,:)=zeros(1,7)
end
if size(pts,1)==2
    pts(3,:)=zeros(1,7)
end
accMax=[15,7.5,10,12.5,15,20,20]';
system("ControlBlock_NEWER.slx");