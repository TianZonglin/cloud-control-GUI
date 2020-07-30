#include "include/scalarimage.h"
#include "include/pointcloud.h"
#include "include/sparsematrix.h"
#include "include/vis.h"
#include "include/skelft.h"

using namespace std;




ScalarImage::ScalarImage(int wd,int ht): image_max(0)
{
	image = new FIELD<float>(wd,ht);
	*image = 0;
	
	certainty = new FIELD<float>(wd,ht);
	*certainty = 0;
}




ScalarImage::~ScalarImage()
{
	delete image;
	delete certainty;
}




void ScalarImage::interpolateDistMatrix(const PointCloud& pc,float delta)
{
	image_max = 0;	
	for(int i=0,Y=image->dimY();i<Y;++i)
	{
		for(int j=0,X=image->dimX();j<X;++j)
		{
			Point2d pix(j,i);									//current point to interpolate to
			float cert;
			float val = pc.interpolateDistMatrix(pix,cert,delta);
			image->value(j,i) = val;
			if (image_max<val) image_max = val;
			certainty->value(j,i) = cert;
		}
	}
}



void ImageInterpolator::shepard(const PointCloud& cloud,const float* point_data,float* out_image,float rad_blur,bool color_mode)
{
	float rad_max  = cloud.avgdist;							//Image is empty further away from the points than the cloud's average-dist..
	int   winSize  = cloud.fboSize;

	float* output = new float[winSize*winSize];

	//ALEX: Experimental code for doing the Shepard interpolation in CUDA - doesn’t work yet.
	//CUDA: Pass DT to CUDA
	//      Pass point_data[] to CUDA, as rendered 2D array of floats
	//initShepard(cloud,point_data);
	//::shepard(output,rad_blur,rad_max);
	
	vector<int> nn;
	const int NN_MAX = 200;
	nn.reserve(NN_MAX);
	
	for(int i=0,offs=0;i<winSize;++i)
	   for(int j=0;j<winSize;++j,++offs)
	   {
		  
		  const Point2d p(j,i);								//The current pixel to shade
		  
		  float dt = cloud.siteDT[offs];					//Distance of current pixel to closest cloud point
		  
		  if (dt>rad_max)									//Too far away from the cloud? Nothing to draw, no points there
		  {
		    out_image[4*offs+3] = 0;
			continue;
		  }
		  
		  //ALEX: Here starts the code to do Shepard on the CPU	

		  float rad  = dt + rad_blur;						//Consider the contribution of all cloud-points within this radius 
		  float rad2 = rad*rad;								//- dt ensures that at least ONE cloud-point will contribute;
															//- rad_blur controls how much blurring (of the cloud's data) we want here
		  
		  nn.clear();										//Get cloud points within 'rad' to current pixel
		  int NN = cloud.searchR(p,rad,NN_MAX,nn);			
									
		  float wsum = 0;									//Compute the Shepard interpolation (of all neighbors) at current pixel: 
		  float val  = 0;
		  for(int j=0;j<NN;++j)
		  {
		    int   pj = nn[j];
		    float r2 = p.dist2(cloud.points[pj]);			//Compute influence of j-th neighbor to current pixel
						
		    float w  = exp(-5*r2/rad2);				
		    wsum    += w;									//Needed to normalize the interpolation, later on			
		    val     += w*point_data[pj];					//Contribution of j-th neighbor
		  }
		  if (wsum<1.0e-6) wsum=1;							//Just in case that we have no contributor (shouldn't happen..)
		   										//Normalize interpolation
		  
		  //ALEX: enable the line below, delete all code above when CUDA Shepard works
		  //float val = output[offs];

		  float r,g,b;										//Generate the final color image:
		  float2rgb(val,r,g,b,color_mode);
		  out_image[4*offs+0] = r;
		  out_image[4*offs+1] = g;
		  out_image[4*offs+2] = b;
		  out_image[4*offs+3] = 1-dt/rad_max;				//Alpha encodes the distance to cloud	
	   }		

	delete[] output;
}



/*
void ImageInterpolator::computeLabelMixing()				//Compute smooth label-mixing image 'tex_mixing' from the per-point label-mixing metric stored in the cloud.
{
	float rad_max  = cloud->avgdist;						//Image is empty further away from the points than the cloud's average-dist..
	float rad_blur = label_mix_averaging;					//How sharp (or blurred, i.e. averaged) the label-mixing image should be

	float* img = new float[winSize*winSize*4];
	int     NL = cloud->numLabels();
	
	vector<int> nn;						
	for(int i=0,offs=0;i<winSize;++i)
	   for(int j=0;j<winSize;++j,++offs)
	   {
		  const Point2d p(j,i);								//The current pixel to shade
		  
		  float dt = cloud->siteDT[offs];					//Distance of current pixel to closest cloud point
		  if (dt>rad_max)									//Too far away from the cloud? Nothing to draw, no points there
		  {
		    img[4*offs+3] = 0;
			continue;
		  }
		  
		  float rad  = dt + rad_blur;						//Consider the contribution of all cloud-points within this radius 
		  float rad2 = rad*rad; 
		  
		  nn.clear();										//Get cloud points within 'rad' to current pixel
		  int NN = cloud->searchR(p,rad,200,nn);		
									
		  hash_map<int,float> lbls;
		  for(int j=0;j<NN;++j) lbls.insert(make_pair(cloud->point_scalars[nn[j]],1.0e+6));
		  for(int j=0;j<NN;++j)
		  {
		    int   pj = nn[j];
		    float r2 = p.dist2(cloud->points[pj]);			//Compute influence of j-th neighbor to current pixel
			int   lj = cloud->point_scalars[pj];
			hash_map<int,float>::iterator it = lbls.find(lj);
			if (it->second>r2) it->second = r2;
		  }

		  float val = 0;
		  for(hash_map<int,float>::const_iterator it=lbls.begin();it!=lbls.end();++it)
		  {
		     float r2 = it->second;
		     val += exp(-5*r2/rad2);
		  }
		  val /= NL;
		  
		  float r,g,b;										//Generate the final color image:
		  float2rgb(val,r,g,b,true);
		  img[4*offs+0] = r;
		  img[4*offs+1] = g;
		  img[4*offs+2] = b;
		  img[4*offs+3] = 1-dt/rad_max;						//Alpha encodes the distance to cloud (just as tex_mask)	
	   }	


	glEnable(GL_TEXTURE_2D);									//Store generated image in texture 'tex_mixing'	
	glBindTexture(GL_TEXTURE_2D,tex_mixing);
	glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,winSize,winSize,0,GL_RGBA,GL_FLOAT,img);
	glDisable(GL_TEXTURE_2D);		
	delete[] img;
}
*/





