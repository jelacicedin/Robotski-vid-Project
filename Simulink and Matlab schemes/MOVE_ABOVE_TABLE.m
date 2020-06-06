%% Forming robot

robot = importrobot('panda2.urdf');
robot.DataFormat = 'column';


Kp = 2;
Kd = 0.1;


%% Loading Python script
warning('off')
clear classes
mod = py.importlib.import_module('petriDish');
%% Add here the other python files we'll be using

%% Moving manipulator into initial photo-op position

runForDuration = 5;

x = [0.08; 0.25];
y = [0.2; -0.24];
z = [0.6; 0.6];

th1 = [0; 0];
th2 = [0; 0];
th3 = [0; 0];

[xn, yn, zn, ~,~,~] = fixedInterpolationOperational(x,y,z, th1, th2, th3, runForDuration);

run('InterpolationIkinePD2019.slx')

pause(30);          % Take photo on spacebar





