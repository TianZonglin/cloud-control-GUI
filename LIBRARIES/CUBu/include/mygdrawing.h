#pragma once

#include "include/gdrawing.h"

class MyGraphDrawing : public GraphDrawing
{
public:

						GraphDrawing();
virtual				   ~GraphDrawing();	
const MyGraphDrawing&	operator=(const MyGraphDrawing& rhs);
void					draw() const;
void					normalize(const Point2d& dim,float border);
void					saveTrails(const char* fn) const;
bool					readTrails(const char* fn);


void					resample(Polyline&,float delta,float jitter);

std::vector<Row>		drawing;			//edges from i-th point to all points > i
DrawingOrder			draw_order;			//edges sorted by some (drawing) order
float					val_min,val_max;	//range of edge values
};



