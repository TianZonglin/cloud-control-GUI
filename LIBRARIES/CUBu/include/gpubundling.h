#pragma once 

//Implementation header: Should only be used by implementation-level code, not client code
//
//

#include "include/cpubundling.h"
#include "include/Point3d.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>

//!!!Probably move all this to GPUBundling class decl

struct GLBundlingPoint
{
    float2 coord;
    unsigned char rgba[4];
};


typedef float4 BundlingPoint;
                                                                //An edge sample-point, with additional data for bundling
																//x: pixel-coordinate of the point
																//y: pixel-coordinate of the point
																//z: edge profile (used for controlling the shape of the bundling)
																//w: original edge tangent (0..2*PI)


const int	NTHREADS_X = 32;									//Max #threads we can spawn on the current GPU (x direction)
const int	NTHREADS_Y = 16;									//Max #threads we can spawn on the current GPU (y direction)	
const int	NTHREADS   = NTHREADS_X*NTHREADS_Y;					//Max #threads we can spawn on the current GPU (in a thread-block)

const int	EDGE_PROFILE_SIZE = 1024;
const int	MAX_KERNEL_LENGTH = 2*128+1;						//Maximum kernel length being ever used (must be an odd number)



extern "C" void setConvolutionKernel(float* h_Kernel,int kernel_size,int kernel_radius);

extern "C" void random_init(curandState* d_states);

extern "C" void initializeSiteLocations(CPUBundling::DENSITY_ESTIM,cudaArray* a_input,float* d_outSitemap,unsigned int* d_Count,BundlingPoint* d_inpoints,int npoints,float value,int imageW,int imageH);

extern "C" void initializeSiteMap(unsigned int* d_siteMap,BundlingPoint* d_points,int npoints,int imageW,int imageH);


extern "C" void convolutionGPU(float* d_outDensity,cudaArray* a_inpSitemap,int imageW,int imageH);

extern "C" void advectSites(BundlingPoint* out_points,BundlingPoint* in_points,int npoints,float* a_Src,unsigned int* d_siteMap,int imageW,int imageH,
						    float h,bool tangent,float rep_strength);

extern "C" void resample(BundlingPoint* outpoints,int& n_outpoints,int* outedges,BundlingPoint* in_points,int n_inpoints,int* in_edges,int* h_edges,int n_edges,float delta,curandState* d_rndstates,float jitter,
						 float* d_edgeProfile);

extern "C" void smoothLines(BundlingPoint* out_points,BundlingPoint* in_points,int npoints,float t,float h,float filter_kernel,int niter);

extern "C" void computeShading(float* d_Output,cudaArray* a_Src,int imageW,int imageH,const Point3d&,float radius,bool tube_style);

extern "C" void drawing2GL(GLBundlingPoint* gl_points, BundlingPoint* in_points, int n_inpoints, int* in_edges, int n_edges);


