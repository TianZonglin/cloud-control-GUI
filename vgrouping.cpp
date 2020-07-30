#include "include/vgrouping.h"
#include "include/pointcloud.h"
#include "include/grouping.h"
#include "include/io.h"
#include "include/field.h"


using namespace std;





VisualGrouping::VisualGrouping():grouping(0),cloud(0),cushion_type(CUSHION_BORDER),texids(0),cushion_shading_thickness(20)
{
}

VisualGrouping::~VisualGrouping()
{
   clear();
}


void VisualGrouping::draw(Cushion* c,float alpha,bool cushion_coloring,bool tex_interp)
{													//Draw given cushion
   float r=0,g=0,b=1;								//Base color for cushions (if we don't use color-mapping)
   if (cushion_coloring)							//If we use coloring, color-map the scalar-value for this cushion
	  float2rgb(c->scalar_value,r,g,b,true);		//Otherwise, just use black as cushion color
   glColor4f(r,g,b,alpha);

   setTexture(c->tex_id,tex_interp);				//Bind to the cushion's texture

   glBegin(GL_QUADS);								//Draw the cushion
   glTexCoord2f(0,0); glVertex2f(0.0,0.0); 
   glTexCoord2f(1,0); glVertex2f(cloud->fboSize,0.0); 
   glTexCoord2f(1,1); glVertex2f(cloud->fboSize,cloud->fboSize); 
   glTexCoord2f(0,1); glVertex2f(0.0,cloud->fboSize); 		
   glEnd();
}


void VisualGrouping::clear()						//Release all memory and other resources allocated by this
{
   for(Group2Tex::const_iterator it=group2tex.begin();it!=group2tex.end();++it)
      delete[] it->second.height;					//Release CPU memory with height maps
   
   if (texids)										//Release GL memory for textures...
   {
     glDeleteTextures(group2tex.size(),texids);
     delete[] texids;								//...and also the CPU-side tex-id array
   }	 
   group2tex.clear();								//No more cushions stored in this
}


void VisualGrouping::init(Grouping* pg)				//Allocate space for storing cushions for all groups in 'pg'
{
   clear();
   grouping = pg;
   cloud    = grouping->cloud;
   
   int NG = grouping->size();
   texids = new GLuint[NG+1];							//Space to generate texture ids			
   glGenTextures(NG+1,texids);						//Generate all required textures
   const int winSize2 = cloud->fboSize*cloud->fboSize;

   float* height = new float[winSize2];
   Cushion c(height,texids[0],0);

   group2tex.insert(make_pair(-1,c));
   for(int i=1;i<NG+1;++i)  
   {
       float* height = new float[winSize2];
	   float  sval   = float(i-1)/(NG-1);
	   Cushion c(height,texids[i],sval);
	   group2tex.insert(make_pair(i-1,c));	
   } 
}


void VisualGrouping::setCushion(int gid,float* data,bool norm)//Set the height profile for cushion 'gid'
{															 //Copy the height data passed by the user into cushion for 'gid'
    Cushion* c = getCushion(gid);
	if (!c) return;	
	memcpy(c->height,data,cloud->fboSize*cloud->fboSize*sizeof(float));	
	c->normalized = norm;
}


void VisualGrouping::makeTextures()					//All cushion data is set: Make corresponding GL textures for them, 
{													//so we can draw them next
   const int winSize2 = cloud->fboSize*cloud->fboSize;
   float* buffer = new float[winSize2*2];

   glEnable(GL_TEXTURE_2D);
   for(Group2Tex::const_iterator it=group2tex.begin(),ie=group2tex.end();it!=ie;++it)
   {
	  const Cushion& c = it->second;
	  bool        norm = c.normalized;
	  const float* cht = c.height;
	  
	  for(int i=0;i<winSize2;++i)
	  {
	    float ht = cht[i];
		float alpha,lum;
				
	    if (ht<0)
		{
		  lum   = 0;
	      alpha = 0;
		}
	    else 
		{
		  if (!norm)								//Height not normalized: Clamp it here to [0,1] and also apply a cushion profile
		  {											//If height is normalized, we assume it encodes already the right profile, whichever that is
		    if (ht>cushion_shading_thickness) ht = 1;
		    else ht = pow(ht/cushion_shading_thickness,0.4f);
		  }
		  		
		  lum   = (cushion_type==CUSHION_BORDER)? 1 : ht;
	      alpha = (cushion_type==CUSHION_BORDER)? 1-ht : 1;
		} 

		buffer[2*i]   = lum;
		buffer[2*i+1] = alpha;
	  }
	  
	  glBindTexture(GL_TEXTURE_2D,c.tex_id);
	  glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,cloud->fboSize,cloud->fboSize,0,GL_LUMINANCE_ALPHA,GL_FLOAT,buffer);
   }
   
   delete[] buffer;
}	


VisualGrouping::Cushion* VisualGrouping::getCushion(int gid)
{
    Group2Tex::iterator it = group2tex.find(gid);
	if (it==group2tex.end())
	{
  	  cout<<"Error: VisualGrouping does not contain group-id "<<gid<<endl;
	  return 0;
	}
	
	return &it->second;
}


Grouping::PointGroup VisualGrouping::groupAtPoint(int gid)
{
  Grouping::PointGroup ret;
  if (gid<0) return ret;

  Cushion* c = getCushion(gid);   
  const float* cht = c->height;
  bool  norm = c->normalized;



  Grouping::PointGroup grp;
  grouping->group(gid,grp);

  for(Grouping::PointGroup::const_iterator it=grp.begin();it!=grp.end();++it)
  {
	  int pid = *it;
	  const Point2d& p = grouping->cloud->points[pid];
	  float ht = cht[cloud->fboSize*int(p.y) + int(p.x)];

	  if (ht<0) continue;											//Outside of any cushion..

	  if (!norm)													//Same height normalization as for rendering
	  {
		if (ht>cushion_shading_thickness) ht = 1;
		else ht = pow(ht/cushion_shading_thickness,0.4f);
	  }
	  
	  if (ht>0)
	  {
		ret.insert(pid);
	  }
  }
	
   return ret;
}



int	VisualGrouping::cushionAtPoint(const Point2d& p) const			//Get ID of tallest cushion at given point (-1 if no cushion there)
{
   float min_ht     = 1.0e6;
   int   cushion_id = -1;
   
   for(Group2Tex::const_iterator it=group2tex.begin(),ie=group2tex.end();it!=ie;++it)
   {
	  const Cushion& c = it->second;
	  const float* cht = c.height;
	  float ht = cht[cloud->fboSize*int(p.y) + int(p.x)];
	  bool  norm = c.normalized;
	  
	  if (ht<0) continue;											//Outside of any cushion..
	  if (it->first<0) continue;									//In the visual cushion.. not handled yet
	  
	  if (!norm)													//Same height normalization as for rendering
	  {
		if (ht>cushion_shading_thickness) ht = 1;
	    else ht = pow(ht/cushion_shading_thickness,0.4f);
	  }

	  if (ht>min_ht) continue;
	  cushion_id = it->first;
	  min_ht = ht;
   }
   
   return cushion_id;
}





