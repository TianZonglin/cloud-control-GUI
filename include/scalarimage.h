#pragma once

#include "include/field.h"


class PointCloud;
class Point2d;




class ImageInterpolator
{
public:


	static void	shepard(const PointCloud&,const float* point_data,float* out_image,float rad_blur,bool color_mode);

};






class ScalarImage
{
public:
	
			    ScalarImage(int width,int height);			//Ctor: init image
			   ~ScalarImage();								//Dtor: dealloc all owned by this
	void		interpolateDistMatrix(const PointCloud&,float);
	
	FIELD<float>* image;									//The raw scalar image	
	FIELD<float>* certainty;								//Per-point certainty [0..1] for the above
	
	float		image_max;									//Max of the scalar data
};



