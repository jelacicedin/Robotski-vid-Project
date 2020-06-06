the robot should move to the first above the table pose
------------------------------
for the initial petri dish detection
use file petriDish.py
function is called:
(x,y,r)=getPetriDish(image)
or:
(x,y,r)=getPetriDish(image,1)
to enable result printing and image drawing
function returns a tuple containing the x and y coordinates of the found dish and its radius
------------------------------
the robot should now zoom in for a closer up picture of the found petri dish
------------------------------
for the individual bacteria detection
use file bacteriaFinder.py
function is called:
coordinates=findBacteria(image)
or:
coordinates=findBacteria(image,1)
to enable intermediate steps and result printing and drawing
function returns a numpy array of coordinates stacked into rows:
[[x1 y1]
 [x2 y2]
 [x3 y3]
  ...  ]