#pragma once


#include "include/field.h"
#include "include/pointcloud.h"

class PointCloud;
class Image;




class Animator												//Baseclass for all particle animator engines.
{
public:	
	
	virtual        ~Animator();
	virtual void	setCloud(PointCloud*);					//Set the cloud on which the animation is to be done. Initializes an animation.
	virtual void	move() =0;								//Execute one animation step. Moved particles are in PointCloud::moved_points.
	
protected:

					Animator(): cloud(0) {}
	PointCloud*	    cloud;
};




class TargetAnimator : public Animator						//Simple animator, using linear interp between two precomputed 
{															//(src,dst) positions for each site
public:
					TargetAnimator(int nsteps);				//Ctor. #steps to create must be given.
	
	void			move();									//Impl inh
	
protected:
	
int					nsteps;
float				time;	
};



class AdvectAnimator : public Animator						//Particle dynamics animator. Uses a combination of attraction (towards the shape's skeleton)
{															//and repulsion (particle-to-particle) iterative process.
public:

					AdvectAnimator(Image* image,float step,int relax_iters,int tot_iters);
															//Ctor: target-image,advection-step-size,#smoothing-iterations/move,#advect+smooth-iterations/move
				   ~AdvectAnimator();						

void				move();									//Impl inh
void				setCloud(PointCloud*);					//Enh inh
	
protected:
	
float				step;
float				radius;
Image*				image;
short*				FT;
float*				DT;
float*				map;
int					fboSize;
int					iterations;
int					tot_iterations;
float*				sites;
};


