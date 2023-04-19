//Gmsh project created on Thu Jan 19 11:58:52 2023
//Author : Pierre Laurentin

//Greatly inspired of https://github.com/ComputationalDomain/CMesh_rae69ck-il of the "Computational Domain" YouTube channel (https://www.youtube.com/@computational_domain)

//Parameter for a symetric (00XX) NACA airfoil

t = 0.12; // XX=100*t
c = 1; //size of the chord

nbr_pts = 1000; // Warning : this corresponds to the number of points on ONE side of the NACA, hence the total number of points nbr_pts_tot = 2*nbr_pts on the whole airfoil

//sub_div = 4; // Parameter for the Boundary Layer Field, best if unchanged. WARNING : Best if it divides nbr_pts/2

angle = 0; //Angle between the horizontal and the chord

//Specify three different mesh sizes : 

wingFront = 0.0045; //Mesh size on the leading edge
wing = 0.09;  //Mesh size on the airfoil, except for leading edge
wall = 1; //Mesh size close to the walls


frontSide = Floor(nbr_pts/10);

//First the points for the NACA are built, with more points on the leading edge for a more precise description of the curvature.

For k In {0:frontSide} //Points on the upper leading edge

x = (k/nbr_pts)*c;
y=(t/0.2)*(0.2969*Sqrt(x) - 0.1260*(x/c) - 0.3516*(x/c)^(2) + 0.2843*(x/c)^(3) - 0.1036*(x/c)^(4)) ;

Point(k) = {x+1,y,0,wingFront};
Rotate {{0,0,1}, {-1.5,0,0}, -(angle*Pi)/180} { Point{k}; }
EndFor


For k In {frontSide+1:nbr_pts} //Points on the upper airfoil

x = (k/nbr_pts)*c;
y=(t/0.2)*(0.2969*Sqrt(x) - 0.1260*(x/c) - 0.3516*(x/c)^(2) + 0.2843*(x/c)^(3) - 0.1036*(x/c)^(4)) ;

Point(k) = {x+1,y,0,wing};
Rotate {{0,0,1}, {-1.5,0,0}, -(angle*Pi)/180} { Point{k}; }
EndFor

For k In {1:nbr_pts-1 - frontSide} //Points on the lower airfoil
x = (1-(k/nbr_pts))*c;
y=(t/0.2)*(0.2969*Sqrt(x) - 0.1260*(x/c) - 0.3516*(x/c)^(2) + 0.2843*(x/c)^(3) - 0.1036*(x/c)^(4)) ;

Point(nbr_pts + k) = {x+1,-y,0,wing};
Rotate {{0,0,1}, {-1.5,0,0}, -(angle*Pi)/180} { Point{nbr_pts + k}; }
EndFor

For k In {nbr_pts - frontSide:nbr_pts-1} //Points on the lower leading edge

x = (1-(k/nbr_pts))*c;
y=(t/0.2)*(0.2969*Sqrt(x) - 0.1260*(x/c) - 0.3516*(x/c)^(2) + 0.2843*(x/c)^(3) - 0.1036*(x/c)^(4)) ;

Point(nbr_pts + k) = {x+1,-y,0,wingFront};
Rotate {{0,0,1}, {-1.5,0,0}, -(angle*Pi)/180} { Point{nbr_pts + k}; }
EndFor

Line(1) = {0:2*nbr_pts-1,0};

//The following parameters were tuned for the specific mesh size of this problem.
nLeadingEdge = 40;
nVertical = 40;
progVertical = 1/0.80;

nUpper = 60;
nLower = 60;

nWake = 50;
rWake = 1/0.92;

rUpper = 1;
rLower = 1;

//Create the box containing the airfoil : c-type mesh as described in J.Chang et al.


Point(100*nbr_pts + 2) = {-0.5,15,0,wall};
Point(100*nbr_pts + 3) = {2,15,0,wall};
Point(100*nbr_pts + 4) = {20,15,0,wall};
Point(100*nbr_pts + 5) = {20,0,0,wall};
Point(100*nbr_pts + 6) = {20,-15,0,wall};
Point(100*nbr_pts + 7) = {2,-15,0,wall};
Point(100*nbr_pts + 8) = {-0.5,-15,0,wall};
Point(100*nbr_pts + 9) = {0,0,0,wall}; //Center


Circle(100*nbr_pts + 1) = {100*nbr_pts + 2,100*nbr_pts + 9,100*nbr_pts + 8};
Line(100*nbr_pts + 2)={100*nbr_pts + 2,100*nbr_pts + 3};
Line(100*nbr_pts + 3)={100*nbr_pts + 3,100*nbr_pts + 4};
Line(100*nbr_pts + 4)={100*nbr_pts + 4,100*nbr_pts + 5};
Line(100*nbr_pts + 5)={100*nbr_pts + 5,100*nbr_pts + 6};
Line(100*nbr_pts + 6)={100*nbr_pts + 6,100*nbr_pts + 7};
Line(100*nbr_pts + 7)={100*nbr_pts + 7,100*nbr_pts + 8};

pc = Floor(nbr_pts*0.02);

Line(100*nbr_pts + 10) = {pc,100*nbr_pts + 2};
Line(100*nbr_pts + 11) = {nbr_pts,100*nbr_pts + 3};
Line(100*nbr_pts + 12) = {nbr_pts,100*nbr_pts + 5};
Line(100*nbr_pts + 13) = {nbr_pts,100*nbr_pts + 7};
Line(100*nbr_pts + 14) = {2*nbr_pts-pc,100*nbr_pts + 8};

Split Curve{1} Point{pc,2*nbr_pts-pc};
Split Curve{100015} Point{pc,nbr_pts};

//The following commands were "handmade" since it is much easier to use the GUI to define the transfinite curve and to access the index of the curves generated by the previous "Split" command.

//What the following command do is basically take an existing curve and place a certain number of points spaced with a geometrical progression specified after the "Using Progression" keyword. Be careful of the curve orientation when using the Transfinite function, the same precautions must e taken with the Transfinite surfaces

//+
Transfinite Curve {100016, 100001} = nLeadingEdge Using Progression 1;
//+
Transfinite Curve {100010, 100014, 100011, 100013, -100004, 100005} = nVertical Using Progression progVertical;
//+
Transfinite Curve {100017, 100002} = nUpper Using Progression rUpper;
//+
Transfinite Curve {100018, 100007} = nLower Using Progression rLower;
//+
Transfinite Curve {100012, 100003} = nWake Using Progression rWake;
//+
Transfinite Curve {100012, -100006} = nWake Using Progression rWake;

//+
Curve Loop(1) = {100014, -100001, -100010, -100016};
//+
Plane Surface(1) = {1};

//+
Curve Loop(2) = {100011, -100002, -100010, 100017};
//+
Plane Surface(2) = {2};

//+
Curve Loop(3) = {100013, 100007, -100014, -100018};
//+
Plane Surface(3) = {3};

//+
Curve Loop(4) = {100004, -100012, 100011, 100003};
//+
Plane Surface(4) = {4};

//+
Curve Loop(5) = {100005, 100006, -100013, 100012};
//+
Plane Surface(5) = {5};

//+
Transfinite Surface {1};
//+
Transfinite Surface {2};
//+
Transfinite Surface {4};
//+
Transfinite Surface {5};
//+
Transfinite Surface {3};

Recombine Surface {1, 2, 3, 5, 4};

Physical Surface(1) = {1,2,3,4,5};

//Give each curve a physical index which will be the one used later in the .prm file to definite boundary conditions on those curves.

Physical Curve(0) = {100016,100017,100018}; //noslip
Physical Curve(1) = {100*nbr_pts + 1}; //inlet
Physical Curve(2) = {100*nbr_pts + 2,100*nbr_pts + 3,100*nbr_pts + 6,100*nbr_pts + 7}; //slip
Physical Curve(3) = {100*nbr_pts + 4,100*nbr_pts + 5}; //outlet 

//Turns the triangles into quadrangles
Mesh.RecombineAll = 1;
Mesh.SubdivisionAlgorithm = 1;
Mesh.Smoothing = 50;

//To generate the mesh, open this file in gmsh and click 2D in the Mesh section. Then export it in .msh format




