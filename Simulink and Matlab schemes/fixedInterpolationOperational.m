function [xint,yint,zint,th1int,th2int,th3int] = fixedInterpolationOperational(x,y,z,th1,th2,th3,time)
% Interpolation of x vector variables for simulation with Ts = 100 Hz

tSamples = time*100;

X = [x';y';z';th1';th2';th3'];

[q,~,~,sampleTime,~] = trapveltraj(X,tSamples);

sampleTime = (0:0.01:time-0.01);


xint = [sampleTime' q(1,:)'];
yint = [sampleTime' q(2,:)'];
zint = [sampleTime' q(3,:)'];
th1int = [sampleTime' q(4,:)'];
th2int = [sampleTime' q(5,:)'];
th3int = [sampleTime' q(6,:)'];



end

