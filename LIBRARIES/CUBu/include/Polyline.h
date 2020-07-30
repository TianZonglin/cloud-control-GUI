#pragma once

#include "include/point2d.h"
#include <vector>


//A sampled curve modeling an edge (bundled or not), with various data stored along it.
//Used mainly for rendering purposes.
//


struct Polyline : std::vector<Point2d>
{
    
struct InterpParams                                     //Parameters for interpolate(). Used to pass all these faster (as a struct)
{
    float   lambda;
    float*  dmap;
    int     dmap_size;
    float   dir_separation;
    float   displ_max;
    bool    displ_absolute;
};
    
    
            Polyline(float v=0,int res=2): value(v) { reserve(res); displ.reserve(res); }
                                                        //Ctor
void        push_back(const Point2d& p)                 //Add a new point at end
            { std::vector<Point2d>::push_back(p); displ.push_back(0); }
    
void        resize(int sz)                              //Resize 'this' to 'sz' points
            { std::vector<Point2d>::resize(sz); displ.resize(sz); }

int         size() const                                //Get #points in this polyline
            { return std::vector<Point2d>::size(); }
    
void		interpolate(const Polyline& p,const InterpParams& params);
                                                        //interpolate 'this' to 'p' using 'params', save result in 'this'

void        resample(float delta,float jitter=0);       //resample this in-place to use new sampling rate
    
float		direction(int i) const;                     //angle of tangent vector at i-th point (0..2PI radians)

float		length() const;                             //length of polyline

float		value;                                      //weight of polyline (application-dependent value)

std::vector<float> displ;                               //displacement (0=none, 1=displ_max) from 'this' to 'p'.
                                                        //Equal to 0 by default. Set only by interpolate(), so far.
};

