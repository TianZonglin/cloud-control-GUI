#pragma once

#include "include/gdrawing.h"

class PointCloud;




class GraphDrawingCloud : public GraphDrawing
{
public:


void		build(const Graph*,PointCloud*);
void		draw() const;										//Over inh
void		draw(float);


private:

float		range;
};




