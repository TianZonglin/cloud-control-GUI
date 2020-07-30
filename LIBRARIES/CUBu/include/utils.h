#pragma once

#include "point2d.h"
#include <math.h>


inline float dir2angle(const Point2d& d)                //Get angle of vector (d,(0,0)), in radians
{                                                       //Return value is in [0,2PI)
	Point2d x = d; x.normalize();

	float cosa = x.x;
	float sina = x.y;
	
	float a;
	if (sina>=0)
		a = acos(cosa);
	else
		a = 2*M_PI - acos(cosa);
	
	return a;
}


inline float dir2angle_axes(const Point2d& d)           //Same as dir2angle, but returns the angle with
{                                                       //respect to the horizontal and vertical directions,
    Point2d x = d; x.normalize();                       //without taking the sign into consideration.
                                                        //Useful for computing a directional colormap
    return acos(fabs(x.x));                             //where the orientation doesn't matter
}


int file_no_lines(const char* fname);                    //Get #lines of text in text file 'fname'
                                                         //Return #lines of text or -1 if error encountered
