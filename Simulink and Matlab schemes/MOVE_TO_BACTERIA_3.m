%% Get all the bacteria and move into them

mod3 = py.importlib.import_module('bacteriaFinder');

dish_image = py.detObjFunc.loadImage('Snaps/snap_1.png');

groups = py.bacteriaFinder.findBacteria(dish_image);

data = double(py.array.array('d',py.numpy.nditer(groups)));

coords = [];

for i = 1:2:length(data)-1
    coords = [coords; data(i) data(i+1)];
end

runForDuration = 10*length(coords);

x_bact = [];
y_bact = [];

x = [x_petri];   % conf for table center
y = [y_petri];
z = [0.2];

for k = 1:length(coords)
    i = coords(k,1);
    j = coords(k,2);
    
    x_t = x_petri+0.1-0.2/(1448-529)*(i-529);
    
    y_t = y_petri+0.1-0.2/(126-1041)*(j-1041);
    
    x = [x; x_t; x_t; x_t];
    y = [y; y_t; y_t; y_t];
    z = [z; 0.2; 0; 0.2];
end

[xn, yn, zn, ~,~,~] = fixedInterpolationOperational(x,y,z, zeros(length(x),1), zeros(length(x),1),zeros(length(x),1), runForDuration);

%% Run the simulation

