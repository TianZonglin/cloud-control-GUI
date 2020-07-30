#include "include/image.h"
#include "include/pointcloud.h"
#include "include/skelft.h"
#include <algorithm>
#include "include/cudawrapper.h"



Image::Image(const char* file):
	   boundaryMax(0),area(0)
{
	image = FIELD<int>::read(file);						//Read image from PGM file
	int nx = image->dimX();
	int ny = image->dimY();
	
	fboSize = skelft2DSize(nx,ny);						//The CUDA buffers are smallest pow(2) that fits this image

	for(int j=0;j<ny/2;++j)
	{
		int j2 = ny-j-1;
		for(int i=0;i<nx;++i)
		{
			int tmp = (*image)(i,j);
			(*image)(i,j) = (*image)(i,j2);
			(*image)(i,j2) = tmp;
		}
	}
														//Compute area of the image (fg pixels)
	for(int i=0;i<nx;++i)
		for(int j=0;j<ny;++j)
			if (!(*image)(i,j)) ++area;

	cudaMallocHost((void**)&boundaryDT,fboSize*fboSize*sizeof(float));
	cudaMallocHost((void**)&boundaryFT,fboSize*fboSize*2*sizeof(short));    
	cudaMallocHost((void**)&boundaryParam,fboSize*fboSize*sizeof(float));	
}



Image::~Image()
{
	delete image; 
	cudaFreeHost(boundaryFT);
	cudaFreeHost(boundaryParam);
	cudaFreeHost(boundaryDT);
}




struct Item
{
	float value;
	Point2d pixel;
	Item(float v,const Point2d& px): value(v),pixel(px) {}
	Item() {}
};



struct index_cmp 
{
	bool operator()(const Item& a, const Item& b) const	//Comparison operator
	{ return a.value > b.value; }
};





void Image::sortPixels(float* DT)
{
	int nx = image->dimX();
	int ny = image->dimY();
	
	vector<Item> vec(area);												//Temporary buffer for sorting pixels in image on their DT values

	int px = 0;															//Fill vec[] with pairs of pixel-coords and their DT-values
	for(int i=0;i<nx;++i)												//(only for foreground pixels)
		for(int j=0;j<ny;++j)
		{
			if ((*image)(i,j)) continue;			
			float dt = DT[j*fboSize+i];
			vec[px++] = Item(dt,Point2d(i,j));
		}

	std::sort(vec.begin(),vec.end(),index_cmp());						//Sort vec[] descendingly on DT
	
	sorted_pixels.resize(area);											//Copy resulting sorted pixel-coords to sorted_pixels[]
	
	for(int i=0;i<area;++i)
		sorted_pixels[i] = vec[i].pixel;
}


void Image::findBoundary()
{
	memset(boundaryParam,0,fboSize*fboSize*sizeof(float));	
	boundaryMax=1;
	
	for(int i=0;i<image->dimX();++i)
		for(int j=0;j<image->dimY();++j)
			if (!(*image)(i,j))											//Foreground pixel:
			{
				if ((*image)(i-1,j) || (*image)(i+1,j) || (*image)(i,j-1) || (*image)(i,j+1)) 
				{														//Neighbor with a background pixel: is thus on the boundary
					boundaryParam[j*fboSize+i] = boundaryMax++;
				}
			}

    skelft2DFT(boundaryFT,boundaryParam,0,0,fboSize,fboSize,fboSize);	//Compute FT of image boundary (over all the image)
	skelft2DDT(boundaryDT,0,0,fboSize,fboSize);							//   Compute DT of image boundary				

}

