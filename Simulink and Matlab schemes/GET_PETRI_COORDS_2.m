%% Get petri dish coordinates in image

image = py.detObjFunc.loadImage('Snaps/snap_0.png');
coords = py.petriDish.getPetriDish(image);


i_dish = double(coords{1});
j_dish = double(coords{2});
r_dish = double(coords{3}); 

x_petri = 0.6/(365-1413)*(i_dish-1413);
y_petri = -0.43/(201-945)*(j_dish-945);

x = [0.25; x_petri];   
y = [-0.24; y_petri];
z = [0.5; 0.2];




[xn, yn, zn, ~,~,~] = fixedInterpolationOperational(x,y,z, th1, th2, th3, runForDuration);
%% Run simulation
%% click spacebar now