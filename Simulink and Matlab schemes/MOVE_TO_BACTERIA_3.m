%% Get all the bacteria and move into them

mod3 = py.importlib.import_module('bacteriaFinder');

dish_image = py.detObjFunc.loadImage('Snaps/snap_1.png');

groups = py.bacteriaFinder.findBacteria(dish_image);