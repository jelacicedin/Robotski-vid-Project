
robot = importrobot('panda2.urdf');
robot.DataFormat = 'column';

%ax = show(robot);


x = 0.2*zeros(1,4)+0.4;
y = 0.15*[-1 -1 1 1];
z = 0.15*[-1 1 -1 1]+0.5;

Kp = 2;
Kd = 0.1;

%hold on
%plot3(x,y,z,'--r','LineWidth',2,'Parent',ax)
%hold off

%%

%%open_system('ControlBlockRobotics.slx')
duration = 30;
[xn,yn,zn, ~,~,~] = fixedInterpolationOperational(x',y',z',zeros(4,1),zeros(4,1),zeros(4,1),duration);
