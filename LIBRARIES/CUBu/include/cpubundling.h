#pragma once


#include "include/Polyline.h"

class GraphDrawing;



class CPUBundling
{
public:

enum DENSITY_ESTIM															//Precision for the kernel density estimation	
{
	DENSITY_EXACT = 0,
	DENSITY_FAST
};

enum EDGE_PROFILE															//Shape (profile) of bundled edges
{
	PROFILE_UNIFORM = 0,													//Uniform: classical FDEB bundling
	PROFILE_HOURGLASS														//Hourglass: bundle terminations are close to original edges (HEB-like)
};	

                CPUBundling(int fboSize);									//Ctor. Allocates framebuffer to do all GPU-related ops
		       ~CPUBundling();	
void			setInput(GraphDrawing*);									//Sets drawing to be bundled next. Drawing must be normalized in [0,fboSize]^2
void			initEdgeProfile(EDGE_PROFILE);								//Sets edge smoothing-profile to use.	
void			bundleCPU();												//Bundle graph in-place on the CPU
void			bundleGPU();												//Bundle graph in-place on the GPU
void			computeDensityShading(GraphDrawing*,float radius,bool shading,bool tube_style);	
																			//Compute density (+ optionally shading) of given graph, save them on CPU in h_densityMap[], h_shadingMap[]

float			h;															//Kernel size (pixels): controls the spatial 'scale' at which we see bundles
float			h_ms;														//Kernel size (pixels) for endpoint bundling
float			eps;														//Advection step as fraction of kernel size ([0,1]): Controls speed of bundling
float			lambda;														//Bundle smoothing ([0,1]): Controls smoothness of bundles
float			lambda_ends;												//End-segment smoothing ([0,1]): Controls smoothness of edge ends (in MS mode)
float			spl;														//Sampling step (pixels) of polylines
int				niter;														//Number of bundling iterations
int				niter_ms;													//Number of meanshift iterations (for edge endpoint clustering)
int				liter;														//Laplacian smoothing iterations. Must be an odd number.
float			smooth_kernel;												//Laplacian smoothing kernel width ([0,1]), fraction of image-size.
float			jitter;														//Jitter factor ([0,1]): Fraction of 'spl' that sample points are jittered along a sampled edge.
DENSITY_ESTIM	density_estimation;											//How we estimate density: exact-per-pixel (slower) or inexact (faster)
EDGE_PROFILE	edge_profile;												//1D function describing bundling strength along an edge
bool			block_endpoints;											//Keep edge endpoints fixed during bundling (classical) or not
bool			verbose;													//Print various messages during execution or not	
float*			h_densityMap;												//Density map (saved to CPU for drawing purposes)
float*			h_shadingMap;												//Shading map (saved to CPU for drawing purposes)
float			densityMax;													//Max value in h_densityMap[]
bool			polyline_style;												//Use polyline-style bundling or classical smooth bundling
bool			tangent;													//Decreases advection in tangent edge-direction or not
float			rep_strength;												//Repulsion strength for directional bundling (in [0,1])

private:

void			bundleEndpointsGPU();
void			bundleEdgesGPU();
void			computeDensity(float rad);
void			computeDirections();
void			drawing2GPU();
void			drawing2CPU();
void			drawing2CPU_raw();
void			endpoints2GPU();
void			endpoints2CPU();
void			smoothCPU(float t,bool end_segs=false);
void            render2GL();
void			applySmooth(const Polyline& pl,Polyline& npl,int j,int L,float t);


GraphDrawing*	drawing;													//Drawing to read from, and write bundling to
int				fboSize;
int				numCtrlPts;													//Total # sample points in the drawing
};



