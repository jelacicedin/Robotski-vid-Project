function A=denavitHartenberg(d,theta,a,alfa)
transZ=eye(4);
transZ(3,4)=d;
rotZ=[cos(theta) -sin(theta) 0 0;
      sin(theta) cos(theta) 0 0;
      0 0 1 0;
      0 0 0 1];
transX=eye(4);
transZ(1,4)=a;
rotX=[1 0 0 0;
      0 cos(alfa) -sin(alfa) 0;
      0 sin(alfa) cos(alfa) 0;
      0 0 0 1];
  A=transZ*rotZ*transX*rotX;