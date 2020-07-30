#include "include/gdrawing.h"
#include "include/graph.h"
#include "include/glwrapper.h"
#include "include/Point3d.h"	
#include "include/Polyline.h"
#include "include/utils.h"
#include <iostream>
#include <iterator>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>



using namespace std;



const int MAX_BUF_SIZE   = 1000000;                             //Max #elements (vertices) we can draw by a single glMultiDrawArrays() call.
const int MAX_LINES_SIZE = 500000;                              //Max #polylines we can draw in a single glMultiDrawArrays() call.




inline float clamp(float f)
{
    return (f>1)? 1 : f;
}

void hsv2rgb(float h, float s, float v, float* rgb)
{
  float m, n, f;
  int i;

  h = 6*h;
  i = floor(h);
  f = h - i;
  if ( !(i&1) ) f = 1 - f; // if i is even
  m = v * (1 - s);
  n = v * (1 - s * f);

  switch (i) {
  case 6:
  case 0: rgb[0]=v; rgb[1]=n; rgb[2]=m; break; 
  case 1: rgb[0]=n; rgb[1]=v; rgb[2]=m; break; 
  case 2: rgb[0]=m; rgb[1]=v; rgb[2]=n; break; 
  case 3: rgb[0]=m; rgb[1]=n; rgb[2]=v; break; 
  case 4: rgb[0]=n; rgb[1]=m; rgb[2]=v; break; 
  case 5: rgb[0]=v; rgb[1]=m; rgb[2]=n; break; 
  } 
}





void GraphDrawing::float2alpha(float value,float& alpha) const
{
		switch(alpha_mode)
		{
		case ALPHA_CONSTANT:
		alpha = 1; 
		break;
		
		case ALPHA_VALUE:
		alpha = value;
		break;

		case ALPHA_INVERSE_VALUE:
		alpha = 1-value;
		break;
		}
}

void GraphDrawing::float2rgb(float value,float* rgb) const	
{
   switch(color_mode)
   {
		case GRAYSCALE:
		rgb[0] = rgb[1] = rgb[2] = value;
		break;
		
		case RAINBOW:
		case DENSITY_MAP:
        case DISPLACEMENT:
		{
		const float dx=0.8f;
		value = (6-2*dx)*value+dx;
		rgb[0] = max(0.0f,(3-(float)fabs(value-4)-(float)fabs(value-5))/2);
		rgb[1] = max(0.0f,(4-(float)fabs(value-2)-(float)fabs(value-4))/2);
		rgb[2] = max(0.0f,(3-(float)fabs(value-1)-(float)fabs(value-2))/2);
		break;
		}

		case INVERSE_RAINBOW:
		{
		const float dx=0.8f;
		value = (6-2*dx)*(1-value)+dx;
		rgb[0] = max(0.0f,(3-(float)fabs(value-4)-(float)fabs(value-5))/2);
		rgb[1] = max(0.0f,(4-(float)fabs(value-2)-(float)fabs(value-4))/2);
		rgb[2] = max(0.0f,(3-(float)fabs(value-1)-(float)fabs(value-2))/2);
		break;
		}
   }	 
}



void GraphDrawing::interpolate(const GraphDrawing& target,float relaxation,float dir_separation,float displ_max,bool displ_absolute)
{
    Polyline::InterpParams p;                                                           //Prepare param-block for Polyline::interpolate
    p.lambda         = relaxation;                                                      //REMARK: Passing all params as an object is faster than
    p.dmap           = densityMap;                                                      //        having a long param-list.
    p.dmap_size      = densityMapSize;
    p.dir_separation = dir_separation;
    p.displ_max      = displ_max;
    p.displ_absolute = displ_absolute;
    
	for(int i=0,NP=drawing.size();i<NP;++i)												//Interpolate all edges in this towards their counterparts in 'target':
	{
		Row&       row  = drawing[i];                                                   //
		const Row& grow = target.drawing[i];

		Row::const_iterator git=grow.begin();
		for(Row::const_iterator it=row.begin();it!=row.end();++it,++git)
		{
			Polyline&        pl = *it->second;
			const Polyline& gpl = *git->second;
			pl.interpolate(gpl,p);
                                                                                        //Interpolate edge 'pl' towards 'gpl'. Result goes into 'gpl'
		}
	}	
}





GraphDrawing::GraphDrawing(): val_min(0),val_max(1),draw_points(false),draw_edges(true),draw_endpoints(false),num_edges(0),line_width(1),global_alpha(1),
							  color_mode(RAINBOW),alpha_mode(ALPHA_CONSTANT),densityMap(0),densityMapSize(0),densityMax(0),use_density_alpha(false),shading(false),
							  amb_factor(0.1),
                              diff_factor(0.7),
                              spec_factor(10),
                              spec_highlight_size(8.0),
                              shadingMap(0),
                              draw_background(false)
{
	srand(time(0));
	
	light = Point3d(0.4,0.4,-1);									//Initialize light to something reasonable
	light.normalize();

	control_point_color[0] = 1; control_point_color[1] = 0; control_point_color[2] = 0;		//Initialize color for control points
	end_point_color[0]     = 0; end_point_color[1]     = 1; end_point_color[2]     = 0;		//Initialize color for end points
	
}


GraphDrawing::~GraphDrawing()
{
	for(DrawingOrder::const_iterator it=draw_order.begin();it!=draw_order.end();++it)
	{
		delete it->second;
	}
}	


const GraphDrawing& GraphDrawing::operator=(const GraphDrawing& rhs)
{
	if (&rhs==this) return *this;
	
	int NR = rhs.numNodes();

	drawing.resize(NR);												//Allocate #nodes
	draw_order.clear();

	for(int i=0;i<NR;++i)											//Add edges for each node: To make sure we don't add an edge twice,
	{																//we only add edges from a node i to all nodes with j>i
		const Row& rhs_row = rhs(i);
		Row&       row     = (*this)(i);
		row.clear();
		
		for(Row::const_iterator it=rhs_row.begin();it!=rhs_row.end();++it)
		{
		   int            key = it->first;	
		   const Polyline* pl = it->second;
		   
		   Polyline* line = new Polyline(*pl);						//Copy ctor

		   Row::const_iterator ins = row.insert(make_pair(key,line)).first;

		   draw_order.insert(make_pair(line->value,ins->second));			//Add edges sorted on increasing 'value' (for later drawing etc)
		}
	}	

	val_min			= rhs.val_min;
	val_max			= rhs.val_max;
	draw_points		= rhs.draw_points;
	draw_edges		= rhs.draw_edges;
	draw_endpoints  = rhs.draw_endpoints;
	num_edges		= rhs.num_edges;
	line_width		= rhs.line_width;
	global_alpha	= rhs.global_alpha;
	color_mode		= rhs.color_mode;
	alpha_mode		= rhs.alpha_mode;
	densityMap		= rhs.densityMap;
	shadingMap		= rhs.shadingMap;
	densityMapSize	= rhs.densityMapSize;
	densityMax		= rhs.densityMax;
	use_density_alpha	= rhs.use_density_alpha;
	shading			= rhs.shading;
	light			= rhs.light;
	amb_factor		= rhs.amb_factor;
	diff_factor		= rhs.diff_factor;
	spec_factor		= rhs.spec_factor;
	spec_highlight_size = rhs.spec_highlight_size;
	scale           = rhs.scale;
	translation     = rhs.translation;
	background_size = rhs.background_size;
	draw_background = rhs.draw_background;
	
	return *this;
}


void GraphDrawing::normalize(const Point2d& dim,float border)				//Normalize graph drawing in the bbox of dim[],
{																			//leaving a white-space of thickness 'border' around.
	Point2d min_p = Point2d(1.0e+7,1.0e+7);
	Point2d max_p = Point2d(-1.0e+7,-1.0e+7);

	for(DrawingOrder::const_iterator it=draw_order.begin();it!=draw_order.end();++it)
	{
		const Polyline& pl = *it->second;
		for(int i=0,N=pl.size();i<N;++i)
		{
			const Point2d& p = pl[i];
			min_p = Point2d(std::min(min_p.x,p.x),std::min(min_p.y,p.y));
			max_p = Point2d(std::max(max_p.x,p.x),std::max(max_p.y,p.y));
		}	
	}
	
	Point2d rng;															//If we do have a background image, the bbox of the
	if (background_size.x)													//drawing to normalize is given by the image, and not
	{																		//the graph (we assume the graph is inside the image)
	  rng = background_size; 
	  min_p = Point2d(0,0);
	}
	else																	//If we don't have a background image, the boox of the
	{																		//drawing to normalize is given by the drawing itself	
	  rng = max_p-min_p;
	}
		
	if (rng.x>rng.y)
		scale  = (1-border)*dim.x/rng.x;
	else
		scale  = (1-border)*dim.y/rng.y;

	translation.x    = (dim.x-scale*rng.x)/2;
	translation.y    = (dim.y-scale*rng.y)/2;

	val_min = 1.0e6; val_max = -val_min;
	for(DrawingOrder::const_iterator it=draw_order.begin();it!=draw_order.end();++it)
	{
		Polyline& pl = *((Polyline*)it->second);
		for(int i=0,N=pl.size();i<N;++i)
		{
			pl[i] = (pl[i]-min_p)*scale + translation;
		}
		
		pl.value = pl.length();										//WARNING: The following 3 lines only make sense if the 'value' of an edge is its length
		val_min = min(val_min,pl.value);
		val_max = max(val_max,pl.value);
	}	  
}	



void GraphDrawing::resample(float delta,float jitter)				//Resample drawing for inter-point distance 'delta' +/- 'jitter'
{
	for(int i=0,NP=numNodes();i<NP;++i)
	{
		Row& row = (*this)(i);

		for(Row::iterator it=row.begin(),ie=row.end();it!=ie;++it)
		{
			Polyline& pl = *it->second;
			pl.resample(delta,jitter);
		}
	}	
}

void GraphDrawing::build(const Graph* m,const PointSet* points)		//Build graph drawing from Graph and point positions
{
	int NR = m->numRows();

	drawing.resize(NR);												//Allocate #nodes
	draw_order.clear();

    num_edges = 0;
	val_min = 1.0e+6;
    val_max = val_min;
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
		   line->push_back((*points)[i]);
		   line->push_back((*points)[j]);
		   Row::const_iterator ins = drow.insert(make_pair(j,line)).first;
		   draw_order.insert(make_pair(val,ins->second));			//Add edges sorted on increasing 'value' (for later drawing etc)
		   val_min = std::min(val_min,val);
		   val_max = std::max(val_max,val);
		   ++num_edges;
		}
	}
}




void GraphDrawing::drawBackground() const							//Draw background image (if any), properly scaled+translated	
{
	if (!background_size.x) return;									//Do we have a valid background image? If not, nothing to do

	glEnable(GL_TEXTURE_2D);
																	//Set texture parameters: 
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

																	//We will next multiply the texture color with the surface color
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);
	
	glBegin(GL_QUADS);												//Draw the background-image (available in the current texture)
	glTexCoord2f(0,1); glVertex2f(translation.x,translation.y);
	glTexCoord2f(1,1); glVertex2f(background_size.x*scale+translation.x,translation.y);
	glTexCoord2f(1,0); glVertex2f(background_size.x*scale+translation.x,background_size.y*scale+translation.y);
	glTexCoord2f(0,0); glVertex2f(translation.x,background_size.y*scale+translation.y);
	glEnd();
	
	glDisable(GL_TEXTURE_2D);

}

void GraphDrawing::draw() const										//Draw the graph with various color-mapping/shading options
{	
    
    static float vert_buffer[2*MAX_BUF_SIZE];                       //Buffers for 2D vertex coords and colors. These are used to limit
    static unsigned char col_buffer[4*MAX_BUF_SIZE];                //the number of GL calls, and thus draw very large graphs faster.
    static GLint line_first[MAX_LINES_SIZE];
    static GLsizei line_size[MAX_LINES_SIZE];
    GLsizei col_offs=0,vert_offs=0;
    GLsizei line_offs = 0;
    GLsizei line_prev = 0;
    
    if (draw_background) drawBackground();

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
	glLineWidth(line_width);
	glEnable(GL_LINE_SMOOTH);
	glHint(GL_LINE_SMOOTH_HINT,GL_NICEST);
	glShadeModel(GL_SMOOTH);
	glEnable(GL_POINT_SMOOTH);
	glPointSize(1.5);
	
	float rho_max = (densityMax)? *densityMax : 1;					//Get max of density map
        
        glEnableClientState(GL_VERTEX_ARRAY);                           //Enable the GL vertex and color arrays.
        glEnableClientState(GL_COLOR_ARRAY);                            //We'll use these to draw batches of polylines faster
        glVertexPointer(2, GL_FLOAT, 0, vert_buffer);
        glColorPointer(4, GL_UNSIGNED_BYTE, 0 , col_buffer);
        


    
	//for(DrawingOrder::const_reverse_iterator it=draw_order.rbegin();it!=draw_order.rend();++it)
	for(DrawingOrder::const_iterator it=draw_order.begin(),ite=draw_order.end();it!=ite;++it)
	{
		const Polyline& pl = *(it->second);
		int N = pl.size();
		float val = (pl.value-val_min)/(val_max-val_min);			//edge length normalized in [0,1]
		float rgb[3],alpha=1;

		if (color_mode==FLAT)							//1. Determine color of edge:
		{  rgb[0]=rgb[1]=rgb[2] = 0.9;}						//Edge hue = gray
		else	
		if (color_mode==DIRECTIONAL)						//Edge hue = direction
		{
		  Point2d fr  = pl[0],     to  = pl[N-1];						
		  float   frn = fr.norm(), ton = to.norm();
		  if (frn>ton)
		  { Point2d tmp = fr; fr = to; to = tmp; }
		  float h = dir2angle(pl[N-1]-pl[0])/(2*M_PI);
		  float s = pow(val,0.3f);
		  hsv2rgb(h,s,1,rgb);
		}
		else float2rgb(val,rgb);						//Edge hue = length
        
											//2. Determine edge transparency
		float2alpha(1-val,alpha);						//Edge base-alpha maps edge length
		alpha *= global_alpha;							//Modulate above with global transparency
					
		for(int j=0;j<N;++j)							//3. Visit all current edge control points (to draw them)
		{
		   if (draw_edges)							//Compute per-control-point colors only if we draw edges
		   {
			   float t = float(j)/(N-1);					//t: 0..1 over edge (arc-length parameter)
			   t = 0.2+ 0.8*pow(1-2*fabs(t-0.5),0.5);			//t: 0..1..0 over edge (cushion profile)
			   t = val*1 + (1-val)*0.5*t;					//luminance: const (1) for longest edges, cushion for shortest edges
			   float ta = (1-val)*1 + val*t*0.5;				//alpha: const (1) for shortest edges, cushion for longest edges
			   
			   int   offs = int(pl[j].y)*densityMapSize+int(pl[j].x);
			   float nrho = densityMap[offs]/rho_max;
			   alpha *= (use_density_alpha)? pow(nrho,0.3f) : 1;
		
			   float diffuse=1, specular=0;					//Compute shading:
			   if (shading)
			   {
				   float shade = shadingMap[offs];			//shadingMap[] contains dot-prod of surf-normal with light
				   diffuse  = amb_factor + diff_factor*shade; 
				   specular = spec_factor*pow(shade,spec_highlight_size);
			   }
			
			   if (color_mode==DENSITY_MAP)                         	//Encode density value into color, if so desired
			   {
			     float2rgb(nrho,rgb);
			     t = 1;
			   }
			   else if (color_mode==DISPLACEMENT)                   	//Encode amount of point displacement into color, if so desired
			   {
			     float2rgb(pl.displ[j],rgb);
			     //ta = pl.displ[j];
			   }
	 
		
			   t *= diffuse;                                        	//Set final RGBA color of edge sample-point
		
			   col_buffer[col_offs++] = 255*clamp(rgb[0]*t+specular);   	//Copy color of current sample-point
			   col_buffer[col_offs++] = 255*clamp(rgb[1]*t+specular);       //to the GL color array
			   col_buffer[col_offs++] = 255*clamp(rgb[2]*t+specular);
			   col_buffer[col_offs++] = 255*clamp(alpha*ta);
		   }

		   vert_buffer[vert_offs++] = pl[j].x;					//Copy coorss of current sample-point to the
		   vert_buffer[vert_offs++] = pl[j].y;			   		//GL vertex array
		}
		
		bool dump = false;                                      		//
		DrawingOrder::const_iterator itc = it; ++itc;           		//
		if (itc==ite) dump = true;                              		//Is this the last polyline in the graph? Then draw buffers.
		else                                                    		//Do the buffers have space for the next polyline?
		{                                                       		//If not, we must draw them
		  const Polyline& nxtp = *(itc->second);
		  if (vert_offs + 2*nxtp.size() >= MAX_BUF_SIZE) dump = true;
		}
	    
		if (dump)                                               		//We must draw the buffers: 
		{                                                       		//
											//REMARK: Not sure why, but glMultiDrawArrays crashes randomly under Linux.
											//	  The equivalent glDrawArrays works just fine.
		  //if (line_offs) glMultiDrawArrays(GL_LINE_STRIP,line_first,line_size,line_offs);


		  for(int i=0;i<line_offs;++i)
		     if (line_size[i])
		     {
		        if (draw_edges)							//Draw edges? Then enable the color array
		        {
		           glEnableClientState(GL_COLOR_ARRAY);
			   glDrawArrays(GL_LINE_STRIP,line_first[i],line_size[i]);
			}
			if (draw_points)   						//Draw control points? Then disable the color array
			{								//and use a fixed control-point color
			   glDisableClientState(GL_COLOR_ARRAY);
			   glColor3fv(control_point_color);
			   glDrawArrays(GL_POINTS,line_first[i],line_size[i]);
			}   
		     }	
			
		  col_offs  = 0;							//Flush buffers since we've just drawn them		
		  vert_offs = 0;
		  line_offs = 0;
		  line_prev = 0;
		}
		else                                                     		//We don't need to draw the buffers, there's space enough:
		{                                                        		//Just record the offset (start) and length (size) of the
		  line_size[line_offs]  = N;                             		//current polyline in line_first[], line_size[].
		  line_first[line_offs] = line_prev;                     		//GL will use these when calling glMultiDrawArrays().
		  ++line_offs;
		  line_prev += N;
		}
	}										//Next edge	
    
        glDisableClientState(GL_VERTEX_ARRAY);
        glDisableClientState(GL_COLOR_ARRAY);
     
        /*
        //VBO experiment: Construct drawing directly on GPU
        extern int gl_buffer_size;
        extern GLuint gl_buffer;
    
        if (gl_buffer_size)
        {
	   glBindBuffer(GL_ARRAY_BUFFER, gl_buffer);                               //let GL take hold of the shared block
	   glEnableClientState(GL_VERTEX_ARRAY);                                   //from now on, it can read from it (i.e. draw)
	   glEnableClientState(GL_COLOR_ARRAY);
        
	   glVertexPointer(2, GL_FLOAT, 16, 0);
                                                                                //The value 0 in the last arg of glVertexPointer tells that the vertex data
                                                                                //starts at offset 0 the currently-bound buffer
                                                                                glColorPointer(4, GL_UNSIGNED_BYTE,16,(void*)8);                        //Similarly, the 8 in the last arg of glColorPointer tells that the color data
                                                                                //starts at offset 8 in the currently-bound buffer (i.e., after vertex data)
           glDrawArrays(GL_POINTS, 0, gl_buffer_size);

           glDisableClientState(GL_COLOR_ARRAY);
           glDisableClientState(GL_VERTEX_ARRAY);
         }
         */
    
        cout<<"End drawing"<<endl;
    
    
	if (draw_endpoints)                                                         //5. Draw endpoints: We do this at end so they appear _atop_
	{
		glColor3fv(end_point_color);
		glBegin(GL_POINTS);
		
		for(DrawingOrder::const_iterator it=draw_order.begin();it!=draw_order.end();++it)	//all the edges
		{
			const Polyline& pl = *(it->second);
			int N = pl.size();

			glVertex2f(pl[0].x,pl[0].y);
			glVertex2f(pl[N-1].x,pl[N-1].y);
		}
		
		glEnd();
	}
	
	glDisable(GL_BLEND);
	glDisable(GL_LINE_SMOOTH);
	glLineWidth(1.0f);
	glPointSize(1);
	glDisable(GL_POINT_SMOOTH);
}


void GraphDrawing::saveTrails(const char* fn,bool bundled) const				//Save trails (all sample points or just line between endpoints)
{
	FILE* fp = fopen(fn,"w");
	if (!fp) return;

	int lcount = 0;
	for(int i=0,NP=drawing.size();i<NP;++i)
	{
		const Row& row = drawing[i];
		for(Row::const_iterator it=row.begin();it!=row.end();++it)
		{
			fprintf(fp,"%d: ",lcount++);
			const Polyline& pl = *it->second;
			
			int step = (bundled)? 1 : pl.size()-1;
			for(int j=0;j<pl.size();j+=step)
			{
			   const Point2d& p = pl[j];	
			   fprintf(fp,"%f %f ",p.x,p.y);
			}  
			fprintf(fp,"\n");
		}
	}	
	
	fclose(fp);
}



bool GraphDrawing::readBackground(const char* filename)
{
    FILE *inFile;										//File pointer for reading the image
	char buffer[10000];									//Buffer for reading lines from the input file
    int width, height, maxVal, pixelSize;				//Image characteristics from the PPM file

	if(!(inFile=fopen(filename,"rb")))					//Try to open the file for reading in binary mode
		return false;

	fgets(buffer, sizeof(buffer), inFile);				//Read file-type identifier (magic number)
	if ((buffer[0]!='P') || (buffer[1]!='6')) 
	{
	   fclose(inFile);
	   return false;
	}

	if(buffer[2] == 'A')								//See if the file is a RGB or RGBA image:
		pixelSize = 4;									//File contains RGBA pixels
	else
		pixelSize = 3;									//File contains RGB pixels

	do fgets(buffer,sizeof(buffer),inFile);				//Skip possible comment lines
	while (buffer[0] == '#');							

	sscanf(buffer, "%d %d", &width, &height);			//Read image size	

	do fgets(buffer,sizeof(buffer),inFile);				//Skip possible comment lines
	while(buffer[0] == '#');
		
	sscanf(buffer, "%d", &maxVal);						//Read maximum pixel value (usually 255)

	int memSize = width * height * 4 * sizeof(GLubyte);	//Allocate RGBA texture buffer
	GLubyte* theTexture = new GLubyte[memSize];

	for (int i = 0; i < memSize; i++)					//Read data from file into theTexture[]
	{
		if ((i % 4) < 3 || pixelSize == 4)				//Read RGB color bytes and A btye (if any)
			theTexture[i] = (GLubyte)fgetc(inFile);
		else											//RGB image: set A byte as 255 (fully opaque) 
			theTexture[i] = (GLubyte)255; 
    }
    fclose(inFile);

														//Pass the image in theTexture[] to OpenGL
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,width,height,0,GL_RGBA,GL_UNSIGNED_BYTE,theTexture);


    delete[] theTexture;								//Done with the image. The texture now is stored by OpenGL	
		
	background_size.x = width;							//Remember the background image-size (for later drawing)
	background_size.y = height;
	
	return true;
}


bool GraphDrawing::readTrails(const char* fn,bool only_endpoints, int max_edges)            //Reads a graph (in .txt format) into this
{                                                                                           //If the background image option is on, tries to read an associated
    FILE* fp = fopen(fn,"r");                                                               //background image (filename inferred from graph's filename)
	if (!fp) return false;

    int MAX_LINE = 10000;                                                                    //Max length of a line of text. Should be roughly sufficient for a line sampled with 500 points
    
	draw_order.clear();
	
	Polyline* line = 0;
	int NP=0;
	num_edges = 0;
	val_min = 1.0e6; val_max = -val_min;
	char val[MAX_LINE];
		
	for(int lidx=-1;;)                                                                      //Read one more edge from input file:
	{
	   int c = fscanf(fp,"%s",val);
        
	   bool eof = (c!=1);	
	   bool eol = (eof) || (val[strlen(val)-1]==':');
        
	   if (eol)														//beginning of new polyline edge:
	   {
		   if (line)												//Finish currently built polyline
		   {
			   drawing.push_back(GraphDrawing::Row());
			   GraphDrawing::Row& row = drawing[drawing.size()-1];
			   
			   if (only_endpoints)									//If only endpoints of a trail are to be kept, eliminate all other edge points
			   {
				   Point2d first = (*line)[0], last = (*line)[line->size()-1];	
				   line->clear();
				   line->push_back(first);
				   line->push_back(last);
			   }
			   
			   NP += line->size();
			   			   
			   int nidx = 2*lidx+1;
			   GraphDrawing::Row::const_iterator ins = row.insert(make_pair(nidx,line)).first;
			   ++num_edges;
			   
			   float imp   = line->length();
			   line->value = imp;									//Now that the edge is ready, give it an importance
			   			   			   
			   draw_order.insert(make_pair(line->value,ins->second));	//Add edges sorted on increasing 'value' (for later drawing etc)
			   val_min = std::min(val_min,line->value);
		       val_max = std::max(val_max,line->value);
		   }
		   
		   if (eof) break;
           
		   ++lidx;	
		   if (max_edges && num_edges >= max_edges) break;

		   float value = lidx;
		   		   
		   line = new Polyline(value);								//add new line, with user-defined importance value
	   }	
	   else															//adding new point to current polyline:
	   {
		   if (val[0]!='(' && val[0]!='<') 
		   {
		     Point2d p;
		     p.x = atof(val);
		     fscanf(fp,"%s",val);
		     p.y = atof(val);
		     line->push_back(p);
		   }	 
	   }
	}
	
	fclose(fp);
	
	cout<<"Read: "<<num_edges<<" edges, "<<NP<<" control points";
	
	string bkname = fn;
	int dot = bkname.find_last_of('.');
	bkname = bkname.replace(dot+1,3,"ppm");	
	bool ok = readBackground(bkname.c_str());
	if (ok)
	   cout<<", background image";
	cout<<endl;   
	
	return true;
}
	



