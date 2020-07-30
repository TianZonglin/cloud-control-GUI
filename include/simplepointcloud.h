#pragma once

#include "include/pointcloud.h"



class SimplePointCloud
{
public:

						SimplePointCloud(int N);
int						size() const { return points.size(); }
			
std::vector<Point2d>	points;	
std::vector<Color>		colors;
};




