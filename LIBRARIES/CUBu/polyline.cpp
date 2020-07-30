#include "include/Polyline.h"
#include "include/Point3d.h"
#include "include/utils.h"
#include <iostream>


float Polyline::length() const
{
	int NP  = size();
	float l = 0;
	for(int i=1;i<NP;++i)
	   l += (*this)[i].dist((*this)[i-1]);
	return l;
}   

float Polyline::direction(int i) const
{
	if (i==size()-1) --i;
	Point2d dir = (*this)[i+1]-(*this)[i];
	return dir2angle(dir);
}




void Polyline::interpolate(const Polyline& target,const InterpParams& param)
{												//Interpolate this towards 'target'. This is quite SLOW since we need to find point correspondences.
    int   N  = size()-1;
	int   Nt = target.size()-1;
	float L  = length();
	
    float displ_max = (param.displ_absolute)? param.displ_max*param.dmap_size : param.displ_max*target.length();
    
	static Point2d res[4086];					//Buffer to store interpolated polyline
	
	float l = 0;	
	const Point2d* prev = &(*this)[0];
		
	for(int i=1;i<N;++i)						//Displace this' points (except endpoints):
	{
	   const Point2d* crt = &(*this)[i];        //Get current point
	   l += crt->dist(*prev);                   //l = distance of current-point to polyline start
	   float   t = l/L;							//t = arc-length param of i-th point of this (in [0,1])
	   int     j = t*Nt;						//Find corresponding points of target between which 'crt' lies
												//Find target point 'p' corresponding to i-th point in this
	   float   u = t*Nt-j;						//WARNING: This is not 100% correct, it assumes that 'target' is uniformly sampled
	   Point2d p = target[j]*(1-u) + target[j+1]*u;	
	   
       float l_eff      = 1-param.lambda;       //Clamp maximum displacement of each sample point to 'displ_max'. This is the displacement in param space [0..1]
       float dst        = crt->dist(p);         //dst = the actual displacement that the interpolation would want to do
       float displ_eff  = l_eff*dst;            //The actual displacement will be l_eff*dst
       if (displ_eff > displ_max) l_eff = displ_max/dst;
        
	   res[i] = (*crt)*l_eff + p*(1-l_eff);     //Interpolate between 'p' and i-th point of this, using the interpolation param 'l_eff'
        
       displ[i] = l_eff*dst/displ_max;          //Save the current point's normalized displacement [0..displ_max], for vis purposes next
    
	   
	   if (param.dir_separation!=0)				//Separate bundled edges going in different direction with distance 'dir_separation'
	   {
		   Point3d tan = Point3d(crt->x-prev->x,crt->y-prev->y,0);
		   tan.normalize();						//tan: oriented-tangent along edge at point i
		
		   const Point3d z(0,0,1);				
		   Point3d delta = tan.cross(z);		//delta: shift of point i

												//REMARK: max shift should be proportional with edge length L, more precisely with L/fbo_size
		   float fact = pow(1-2*fabs(t-0.5),0.4);
		   
		   res[i] += Point2d(delta.x,delta.y)*param.dir_separation*fact;
	   }
	   prev = crt;	   
	}

	for(int i=1;i<N;++i) (*this)[i] = res[i];	//Replace this by interpolated curve from res[]
}



void Polyline::resample(float delta,float jitter)
{
    int      NP   = size();
    float    crtd = delta;
    Point2d  prev = (*this)[0];								//'prev' is always on segment (i-1,i)
    
    Polyline nl(value,size());                              //the new, resampled, polyline (allocate some initial space so resizing is fast)
    nl.push_back(prev);										//add 1st point of input polyline to resampled one
    
    for(int i=1;i<NP;)										//resample input polyline:
    {
        const Point2d& crt = (*this)[i];
        float newdist = crt.dist(prev);						//distance from last resampled point to i-th input point
        
        if (newdist<crtd)									//i-th input point closer to 'prev' than remaining fraction of delta:
        {													//skip i-th point
            crtd -= newdist;
            prev  = crt;
            ++i;
        }
        else												//i-th input point farther from 'prev' than remaining fraction of delta:
        {
            float t = crtd/newdist;
            prev = prev*(1-t) + crt*t;						//add new point to resampling
            nl.push_back(prev);
            
            float r = (float(rand())/RAND_MAX)*2-1;
            //generate random number in [-1..1]
            crtd = delta*(1+r*jitter);
            //reset delta to whatever we want to achieve
            //use here a noise equal to jitter*delta
        }
    }	
    
    if (crtd<delta)
        nl.push_back((*this)[NP-1]);
    
    *this = nl;                                                //return resampled line to caller
}

