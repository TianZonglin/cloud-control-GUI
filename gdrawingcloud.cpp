#include "include/gdrawingcloud.h"
#include "include/pointcloud.h"
#include "include/graph.h"
#include "include/io.h"
#include "include/Polyline.h"


void GraphDrawingCloud::build(const Graph* m,PointCloud* cloud)		//Build graph drawing from Graph and point cloud
{
	int NR = m->numRows();

	drawing.resize(NR);												//Allocate #nodes
	draw_order.clear();

	num_edges = 0;
	val_min = 1.0e+6;
	val_max = -1.0e+6;
	for(int i=0;i<NR;++i)											//Add edges for each node: To make sure we don't add an edge twice,
	{																//we only add edges from a node i to all nodes with j>i
		const Graph::Row& mrow = (*m)(i);
		Row& drow = (*this)(i);
		
		for(Graph::Row::const_iterator it=mrow.begin();it!=mrow.end();++it)
		{
		   int     j = it->first;
		   if (j<=i) continue;
		   float val = it->second;

		   Polyline* line = new Polyline(val);
		   line->push_back(cloud->points[i]);
		   line->push_back(cloud->points[j]);
		   Row::const_iterator ins = drow.insert(make_pair(j,line)).first;
		   draw_order.insert(make_pair(val,ins->second));			//Add edges sorted on increasing 'value' (for later drawing etc)
		   val_min = std::min(val_min,val);
		   val_max = std::max(val_max,val);
		   ++num_edges;
		}
	}
	
}


void GraphDrawingCloud::draw(float rng)
{
	range = rng;
	draw();
}

void GraphDrawingCloud::draw() const								//Draw this
{	
	GraphDrawing::draw();
	return;

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
	
	
	for(DrawingOrder::const_iterator it=draw_order.begin();it!=draw_order.end();++it)
	{
		const Polyline& pl = *(it->second);
		float val = (range<1.0e-6)? (pl.value-val_min)/(val_max-val_min) : std::min(pl.value/range,1.0f);

		float r[3];
		float2rgb(val,r);	
		float alpha = val;					
		glColor4f(r[0],r[1],r[2],alpha);
	
		glEnable(GL_LINE_SMOOTH);
		glHint(GL_LINE_SMOOTH_HINT,GL_NICEST);
		glBegin(GL_LINE_STRIP);
		for(int j=0,N=pl.size();j<N;++j)
		   glVertex2f(pl[j]);
		glEnd();
		glDisable(GL_LINE_SMOOTH);
	}	
	
	glDisable(GL_BLEND);
}

