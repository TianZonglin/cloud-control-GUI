#pragma once

#include "include/field.h"
#include "include/pointcloud.h"
#include <vector>


class Image
{
public:
	
			    Image(const char*);						//Ctor: read image from input PGM file
			   ~Image();								//Dtor: dealloc all owned by this
	void		sortPixels(float*);
	void		findBoundary();							//Detect+store b/w image-boundary as needed for the DT/FT API
	
	FIELD<int>* image;									//The raw input image
	
	int			area;									//#foreground pixels in the image
	int			fboSize;								//Size (pow 2) of all imaging ops we will do with this image.
														//Typically, this is the smallest pow(2) in which the image fits.
	int			boundaryMax;							//Highest value of boundary-param as computed by findBoundary()
	
	short*		boundaryFT;
	float*		boundaryParam;
	float*		boundaryDT;
	
	std::vector<Point2d> sorted_pixels;					//Pixels in image[] sorted on some (external) criterion
};



