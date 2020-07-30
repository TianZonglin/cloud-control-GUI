#include "include/cpubundling.h"
#include "include/gdrawing.h"
#include "include/gpubundling.h"											//GPU-side for the bundling utilities
#include "include/glwrapper.h"
#include "include/cudawrapper.h"
#include "include/Point3d.h"
#include <vector>
#include <iostream>
#include <stdio.h>

#include <cuda_gl_interop.h>


using namespace std;


#define TIME(call,t)   \
{                   \
   timerStart();    \
   call;            \
   t = timerEnd();  \
}


//!!Move to class:
	const int		MAX_CTRL_POINTS = 20000000;								//Max # control points we can ever use
	const int		MAX_NUM_EDGES   = 2000000;
	float*			h_Kernel;												//CPU-side kernel
	BundlingPoint*	h_points;												//Bundle control points, CPU-side
    cudaArray*		d_densityRead;											//GPU-side buffer for convolution (read, bound to a tex)
    float*			d_densityWrite;											//GPU-side result of convolution  (written to)
	unsigned int*	d_siteMap;
	unsigned int*	d_siteCount;											//GPU-side buffer counting #sites/pixel (for accurate density estimation)
	BundlingPoint*	d_points;												//Bundle control points, GPU-side (read, bound to a tex)
	BundlingPoint*	d_newpoints;											//Bundle control points, GPU-side (written to)
    StopWatchInterface* 	hTimer;
	float*			h_edgeProfile;
	float2*			h_endpoints;
	float*			d_edgeProfile;
	float*			ms_diststart;
	float*			ms_distend;

	int*			h_edges;
	int*			d_edges;
	int*			d_newedges;
	int				numEdges;
	curandState*	d_rndstates;


//VBO
GLuint                gl_buffer;                                //ID of buffer (that's how OpenGL knows the shared memory block it'll draw from)
cudaGraphicsResource* cuda_buffer;                              //This is the ID for the same thing as above, but seen from CUDA
                                                                //Buffer is ONLY written by CUDA, read by GL
int                   gl_buffer_size = 0;
bool                  use_vbo = false;




static void timerStart()
{
	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);
}

static float timerEnd()
{
	sdkStopTimer(&hTimer);
	return sdkGetTimerValue(&hTimer);
}



CPUBundling::CPUBundling(int sz): drawing(0),fboSize(sz)					//Ctor
{
	h				   = 32;												//Kernel size (pixels,>3): controls the spatial 'scale' at which we see bundles
	h_ms			   = 32;												//Kernel size for endpoint bundling
	eps				   = 0.5;												//Advection step as fraction of kernel size ([0,1]): Controls speed of bundling
	lambda			   = 0.2;												//Bundle smoothing ([0,1]): Controls smoothness of bundles
	lambda_ends		   = 0.5;
	spl				   = 15;												//Sampling step (pixels) of polylines
	niter			   = 15;												//Number of bundling iterations
	niter_ms		   = 0;													//Number of meanshift iterations for endpoints
	liter			   = 1;													//Laplacian smoothing iterations. Must be an odd number.
	smooth_kernel	   = 0.05;
	jitter			   = 0.25;												//Jitter factor ([0,1]): Fraction of 'spl' that sample points are jittered along a sampled edge.
	density_estimation = DENSITY_FAST;										//Accuracy of kernel-density estimation (see CUDA code)
	block_endpoints    = true;												//Keep edge endpoints fixed (classical bundling) or not
	verbose            = true;												//Don't print timing messages
	densityMax		   = 0;
	polyline_style	   = false;
	tangent            = false;
	rep_strength	   = 1;

	const int fboSize2 = fboSize*fboSize;

	ms_diststart = new float[MAX_NUM_EDGES];
	ms_distend   = new float[MAX_NUM_EDGES];
	cudaMallocHost(&h_Kernel,MAX_KERNEL_LENGTH*sizeof(float),cudaHostAllocWriteCombined);//Kernel on CPU (allocate a large enough size for all possible kernels)
	cudaMallocHost(&h_points,MAX_CTRL_POINTS*sizeof(BundlingPoint));		//Control points on CPU (allocate a large enough size, pinned)
    cudaChannelFormatDesc floatTex = cudaCreateChannelDesc<float>();
	cudaMallocArray(&d_densityRead,&floatTex,sz,sz);
	cudaMalloc((void**)&d_densityWrite,fboSize2*sizeof(float));
	cudaMalloc((void**)&d_siteCount,fboSize2*sizeof(unsigned int));
	cudaMalloc((void**)&d_points,MAX_CTRL_POINTS*sizeof(BundlingPoint));
    cudaMalloc((void**)&d_newpoints,MAX_CTRL_POINTS*sizeof(BundlingPoint));

    if (use_vbo)
    {
        glGenBuffers(1, &gl_buffer);                                            // VBO: create ID for GL buffer object
        glBindBuffer(GL_ARRAY_BUFFER,gl_buffer);
        glBufferData(GL_ARRAY_BUFFER,MAX_CTRL_POINTS*sizeof(GLBundlingPoint),0,GL_DYNAMIC_DRAW);      // allocate data for buffer; VBO: this'll beed to be float2+float4
        glBindBuffer(GL_ARRAY_BUFFER,0);                                        // unbind buffer from GL; needed so we register it next on CUDA
        checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_buffer,gl_buffer,cudaGraphicsMapFlagsWriteDiscard)); // register buffer with CUDA for writing-only
    }

	cudaMallocHost(&h_edges,MAX_NUM_EDGES*sizeof(int));
	cudaMalloc((void**)&d_edges,MAX_NUM_EDGES*sizeof(int));
	cudaMalloc((void**)&d_newedges,MAX_NUM_EDGES*sizeof(int));
	cudaMalloc((void**)&d_rndstates,NTHREADS*sizeof(curandState));
	cudaMallocHost(&h_edgeProfile,EDGE_PROFILE_SIZE*sizeof(float));
	cudaMalloc((void**)&d_edgeProfile,EDGE_PROFILE_SIZE*sizeof(float));
	cudaMallocHost(&h_endpoints,2*MAX_NUM_EDGES*sizeof(float2));
	cudaMallocHost(&h_densityMap,fboSize2*sizeof(float));
	cudaMallocHost(&h_shadingMap,fboSize2*sizeof(float));
	cudaMalloc((void**)&d_siteMap,fboSize2*sizeof(unsigned int));


	initEdgeProfile(PROFILE_UNIFORM);
	random_init(d_rndstates);												//Init CUDA random generator
	memset(h_densityMap,0,fboSize2*sizeof(float));
	memset(h_shadingMap,0,fboSize2*sizeof(float));

	sdkCreateTimer(&hTimer);
}


CPUBundling::~CPUBundling()													//Dtor
{
	delete[] ms_diststart;
	delete[] ms_distend;
	cudaFreeHost(h_Kernel);
	cudaFreeHost(h_points);
	cudaFreeArray(d_densityRead);
	cudaFree(d_densityWrite);
	cudaFree(d_siteCount);
	cudaFree(d_points);

    if (use_vbo)
    {
        cudaGraphicsUnregisterResource(cuda_buffer);                    // unregister this buffer object with CUDA
        glBindBuffer(1, gl_buffer);                                     // register the buffer with GL...
        glDeleteBuffers(1, &gl_buffer);                                  // ...and ask GL to delete it
        gl_buffer = 0;
    }

	cudaFree(d_newpoints);
	cudaFreeHost(h_edges);
	cudaFree(d_edges);
	cudaFree(d_newedges);
	cudaFree(d_rndstates);
	cudaFreeHost(h_edgeProfile);
	cudaFreeHost(d_edgeProfile);
	cudaFreeHost(h_endpoints);
	cudaFreeHost(h_densityMap);
	cudaFreeHost(h_shadingMap);
	cudaFree(d_siteMap);

	sdkDeleteTimer(&hTimer);
}


void CPUBundling::setInput(GraphDrawing* gd)								//Set graph-drawing we will operate upon
{
   drawing = gd;
}


void CPUBundling::endpoints2GPU()											//Move edge endpoints to GPU
{
	BundlingPoint* h_ptr = h_points;
	for(int i=0,NP=drawing->numNodes();i<NP;++i)
	{
		const GraphDrawing::Row& row = (*drawing)(i);
		for(GraphDrawing::Row::const_iterator it=row.begin(),ie=row.end();it!=ie;++it)
		{
			const Polyline& pl = *it->second;
			for(int j=0,NP=pl.size();j<NP;j+=NP-1,++h_ptr)
			{
				h_ptr->x = pl[j].x;											//One more endpoint..
				h_ptr->y = pl[j].y;
				h_ptr->z = 1;												//We want full bundling for endpoints
																			//We don't care about h_ptr->w, not used for endpoint-bundling
			}
		}
	}

	numCtrlPts = h_ptr - h_points;
	cudaMemcpy(d_points,h_points,numCtrlPts*sizeof(BundlingPoint),cudaMemcpyHostToDevice);
																			//Pass endpoints to GPU for bundling
}


void CPUBundling::drawing2GPU()												//Copy 'drawing' to GPU
{
	numEdges=0;

	float vmin = drawing->val_min, vmax = drawing->val_max;

	BundlingPoint* h_ptr = h_points;
	for(int i=0,NP=drawing->numNodes();i<NP;++i)
	{
		const GraphDrawing::Row& row = (*drawing)(i);
		for(GraphDrawing::Row::const_iterator it=row.begin(),ie=row.end();it!=ie;++it,++h_ptr)
		{
			const Polyline& pl = *it->second;
			float nv = pl.value/vmax;										//normalized edge-length [0..1] over entire trail-set

			h_edges[numEdges++] = h_ptr-h_points;							//store offset of edge-start in point-vector

			for(int j=0,NP=pl.size();j<NP;++j,++h_ptr)
			{
				h_ptr->x = pl[j].x;
				h_ptr->y = pl[j].y;
				int   SZ = (NP>1)? NP-1:1;
				int pidx = int(j*(EDGE_PROFILE_SIZE-1)/SZ);					//apply edge profile
				h_ptr->z = h_edgeProfile[pidx];								//z coordinate: edge profile
                h_ptr->w = nv; //!!!pl.direction(j); 								//w coordinate: edge-tangent (angle in 0,2*M_PI)
			}

			h_ptr->x = h_ptr->y = -1;										//Add end-of-polyline marker
		}
	}

	numCtrlPts = h_ptr - h_points;

	cudaMemcpy(d_points,h_points,numCtrlPts*sizeof(BundlingPoint),cudaMemcpyHostToDevice);
																			//pass ctrl-points to GPU
	cudaMemcpy(d_edges,h_edges,numEdges*sizeof(int),cudaMemcpyHostToDevice);//pass edge-polylines to GPU
}



void CPUBundling::drawing2CPU()												//Copy drawing from GPU to 'drawing'
{
	cudaMemcpy(h_points,d_points,numCtrlPts*sizeof(BundlingPoint),cudaMemcpyDeviceToHost);
																			//Get ctrl-points from GPU into 'h_points'
	BundlingPoint* h_ptr = h_points;
	for(int i=0,NP=drawing->numNodes();i<NP;++i)							//Put back points into 'drawing'
	{																		//We only copy the x,y fields of BundlingPoint, since we assume
		GraphDrawing::Row& row = (*drawing)(i);								//that the bundling didn't change the others
		for(GraphDrawing::Row::iterator it=row.begin(),ie=row.end();it!=ie;++it)
		{
			Polyline& pl = *it->second;
			++h_ptr;														//Take care not to overwrite endpoints of polylines
			for(int j=1,NP=pl.size()-1;j<NP;++j,++h_ptr)
			{
				pl[j].x = h_ptr->x;
				pl[j].y = h_ptr->y;
			}
			h_ptr += 2;														//Skip endpoint and end-of-polyline marker
		}
	}
}


void CPUBundling::endpoints2CPU()											//Move clustered edge endpoints back to CPU,
{																			//as 2nd..last-1 points of their edges
	cudaMemcpy(h_points,d_points,numCtrlPts*sizeof(BundlingPoint),cudaMemcpyDeviceToHost);

	float2* endp = h_endpoints;

	BundlingPoint* h_ptr = h_points;
	for(int i=0,NP=drawing->numNodes();i<NP;++i)							//For all graph drawing's edges:
	{
		GraphDrawing::Row& row = (*drawing)(i);
		for(GraphDrawing::Row::iterator it=row.begin(),ie=row.end();it!=ie;++it)
		{
			Polyline& pl = *it->second;
			int NP = pl.size();

			Point2d first(h_ptr->x,h_ptr->y);								//Get MS point for edge-start
			++h_ptr;
			Point2d last(h_ptr->x,h_ptr->y);								//Get MS point for edge-end
			++h_ptr;

			*endp++ = make_float2(pl[0].x,pl[0].y);							//Save original edge-start
			*endp++ = make_float2(pl[NP-1].x,pl[NP-1].y);					//Save original edge-end

			pl[0]    = first;												//Set 1st edge-point to 'first'
			pl[NP-1] = last;												//Set last edge-point to 'last'
		}
	}
}



void CPUBundling::drawing2CPU_raw()											//Copy drawing from GPU to 'drawing'. If needed, insert back original edge-ends
{
	cudaMemcpy(h_points,d_points,numCtrlPts*sizeof(BundlingPoint),cudaMemcpyDeviceToHost);

	bool end_meanshift = niter_ms>0;										//Have we done MS on endpoints? If so, we must put original ends back
	float2* endp = h_endpoints;												//Get ctrl-points from GPU into 'h_points'
	BundlingPoint* h_ptr = h_points;

	int K=0;
	for(int i=0,NP=drawing->numNodes();i<NP;++i)							//Put back points into 'drawing'
	{
		GraphDrawing::Row& row = (*drawing)(i);
		for(GraphDrawing::Row::iterator it=row.begin(),ie=row.end();it!=ie;++it)
		{
			Polyline& pl = *it->second;
			pl.clear();

			if (end_meanshift)												//If we used MS on endpoints, add original
			{																//edge startpoint first (MS thereof will follow)
				Point2d start(endp->x,endp->y);
				pl.push_back(start);
				ms_diststart[K] = Point2d(h_ptr->x,h_ptr->y).dist(start);	//Record distance from startpoint to 1st MS-point
				++endp;														//(we'll use it for end-segment smoothing)
			}

			for(;h_ptr->x>=0;++h_ptr)										//Copy bundling of edge (from GPU)
			{
				pl.push_back(Point2d(h_ptr->x,h_ptr->y));
			}

			if (end_meanshift)												//If we used MS on endpoints, add original
			{																//edge endpoint last (after MS thereof)
				Point2d end(endp->x,endp->y);
				ms_distend[K++] = pl[pl.size()-1].dist(end);				//Record distance from startpoint to last MS-point
				pl.push_back(end);
				++endp;														//(we'll use it for end-segment smoothing)
			}

			++h_ptr;														//Skip end-of-polyline marker
		}
	}
}




void CPUBundling::bundleGPU()												//Main entry point (GPU bundling)
{
	if (niter_ms)															//1. Do MS on edge endpoints (if needed)
	   bundleEndpointsGPU();

	bundleEdgesGPU();														//2. Bundle edges (KDEEB)

	if (niter_ms)															//3. If MS was done on edges, resample+smooth
	{																		//   edge terminations
		drawing->resample(spl);												//Resample needed for the (orig-end,MS) line-segments, do it on CPU
		smoothCPU(lambda_ends,true);										//Smooth needed to remove curve kinks around MS points
	}

    if (use_vbo) render2GL();                                               //VBO
}



void CPUBundling::render2GL()
{
    if (!use_vbo) return;

    GLBundlingPoint*               gl_points;                                 //VBO: this'll be a pointer to float2+float4
    size_t num_bytes;

    checkCudaErrors(cudaGraphicsMapResources(1,&cuda_buffer,0));              // CUDA gets hold of the shared block...
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&gl_points, &num_bytes, cuda_buffer));


    drawing2GL(gl_points, d_points, numCtrlPts, d_edges, numEdges);           //Ask CUDA to copy current graph-drawing it stores to the GL-formatted shared block

    ::gl_buffer_size = numCtrlPts;

    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_buffer, 0));          //...unmap shared block, CUDA is done with it
}



void CPUBundling::bundleEndpointsGPU()
{
	float h_fact = pow(2.0f/h_ms,1.0f/niter_ms);							//Bundling refinement ([0,1]): Controls coarse-to-fine bundling.
																			//The default setting ensures that we have a nice decreasing kernel-size from 'h' until
																			//a small kernel (2.0) within our 'niter' iterations.
	float h_kern = h_ms;													//Make local copy of kernel-size, since we'll shrink this

	endpoints2GPU();														//Move edge endpoints to GPU
	for(int i=0;i<niter_ms;++i)
	{
		computeDensity(h_kern);												//Compute density map using a kernel-radius 'h' (GPU) <- d_newpoints
		advectSites(d_newpoints,d_points,numCtrlPts,d_densityWrite,0,fboSize,fboSize,h_kern*eps,false,0);
		std::swap(d_points,d_newpoints);
		h_kern *= h_fact;
	}
	endpoints2CPU();														//Add clustered edge endpoints to CPU drawing (as 2nd..last-1 edge points)
}


void CPUBundling::computeDensityShading(GraphDrawing* gd,float radius,bool shading,bool tube_style)
{																			//Compute density (+optionally shading) of given graph, save these to h_densityMap,h_shadingMap
																			//Useful for postprocessing purposes e.g. shading

	setInput(gd);															//Work on 'gd' from now on

	drawing2GPU();
	computeDensity(radius);													//Evaluate density map of current graph. Needs to be done, since we have no guarantee what the density map currently is.
	cudaMemcpy(h_densityMap,d_densityWrite,fboSize*fboSize*sizeof(float),cudaMemcpyDeviceToHost);

	densityMax = 0;															//Compute max density (for visualization purposes). Should go to CUDA really..
	for(float* rhop=h_densityMap,*rhoe=h_densityMap+fboSize*fboSize;rhop<rhoe;++rhop)
	   if (*rhop>densityMax) densityMax = *rhop;

    if (shading)
	{																		//Evaluate shading from density gradient on the GPU
		computeShading(d_densityWrite,d_densityRead,fboSize,fboSize,gd->light,radius,tube_style);
		cudaMemcpy(h_shadingMap,d_densityWrite,fboSize*fboSize*sizeof(float),cudaMemcpyDeviceToHost);
	}
}



void CPUBundling::bundleEdgesGPU()											//Perform the edge bundling (using only the GPU)
{
	if (!niter) return;

	float s_kern = smooth_kernel*fboSize;

	float rT,cgT,sT,cT,aT,gcT,iT,tT=0;
	unsigned int tPoints = 0;


	float h_fact = pow(2.0f/h,1.0f/niter);									//Bundling refinement ([0,1]): Controls coarse-to-fine bundling.
																			//The default setting ensures that we have a nice decreasing kernel-size from 'h' until
																			//a small kernel (2.0) within our 'niter' iterations.
	float h_kern = h;														//Make local copy of kernel-size, since we'll shrink this

	TIME(drawing2GPU(),cgT);												//Move (resampled) drawing to GPU -> d_points (only once)

	for(int i=0;i<niter;++i)												//Iterate (KDEEB method):
	{
	   if (i==0 || !polyline_style)
	   {
	   int NN;
	   TIME(resample(d_newpoints,NN,d_newedges,d_points,numCtrlPts,d_edges,h_edges,numEdges,spl,d_rndstates,jitter,d_edgeProfile),rT);
	   numCtrlPts = NN;
	   }
	   std::swap(d_edges,d_newedges);
	   std::swap(d_points,d_newpoints);

	   TIME(computeDensity(h_kern),cT);										//Compute density map using a kernel-radius 'h' (GPU) <- d_points

	   if (tangent) computeDirections();									//If we do directional bundling, compute a site-map

	   TIME(advectSites(d_newpoints,d_points,numCtrlPts,d_densityWrite,d_siteMap,fboSize,fboSize,h_kern*eps,tangent,rep_strength),aT);
																			//Advect current drawing, one step, upstream in its density gradient (GPU) -> d_points
	   TIME(smoothLines(d_points,d_newpoints,numCtrlPts,lambda,spl,s_kern,liter),sT);	//Laplacian smoothing of graph-drawing edges (GPU) -> d_points

	   h_kern *= h_fact;													//Decrease kernel size (coarse-to-fine bundling)

	   iT = rT+cT+aT+sT; tT += iT;
	   tPoints += numCtrlPts;
	   if (verbose)
		  cout<<"Iteration: "<<i<<": time: "<<iT<<"  resample "<<rT<<"  smooth: "<<sT<<"  splat: "<<cT<<"  advect: "<<aT<<"; #pts: "<<numCtrlPts<<endl;
	}

	TIME(drawing2CPU_raw(),gcT);											//Move back drawing to CPU (for resampling) <- d_points
	tT += cgT + gcT;

	if (verbose)
	   cout<<"Total time (GPU): "<<tT<<"msecs, iter avg time: "<<tT/niter<<" msecs, #iters: "<<niter<<", kernel: "<<h<<" avg #points: "<<tPoints/niter<<endl;
}



void CPUBundling::bundleCPU()												//Perform the bundling (using partly the CPU, partly the GPU)
{
	if (!niter) return;

	float h_fact = pow(2.0f/h,1.0f/niter);									//Bundling refinement ([0,1]): Controls coarse-to-fine bundling.
																			//The default setting ensures that we have a nice decreasing kernel-size from 'h' until
																			//a small kernel (2.0) within our 'niter' iterations.

	float h_kern = h;														//Make local copy of kernel-size, since we'll shrink this

	float s_kern = smooth_kernel*fboSize;

	float tT=0;
	unsigned int tPoints = 0;

	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);
	for(int i=0;i<niter;++i)												//Iterate (KDEEB method):
	{
		drawing->resample(spl);												//Resample drawing on the CPU
		drawing2GPU();														//Copy to GPU

		computeDensity(h_kern);												//Compute density map using a kernel-radius 'h' (GPU) <- d_newpoints

 	    if (tangent) computeDirections();

		advectSites(d_newpoints,d_points,numCtrlPts,d_densityWrite,d_siteMap,fboSize,fboSize,h_kern*eps,tangent,rep_strength);
																			//Advect current drawing, one step, upstream in its density gradient (GPU) -> d_points
		smoothLines(d_points,d_newpoints,numCtrlPts,lambda,spl,s_kern,liter);	//Laplacian smoothing of graph-drawing edges (GPU) -> d_newpoints

		drawing2CPU();														//Copy back new drawing to CPU

		h_kern *= h_fact;
		tPoints += numCtrlPts;
	}
	tT = sdkGetTimerValue(&hTimer);

	if (verbose)
	   cout<<"Total time (CPU): "<<tT<<" msecs, iter avg time: "<<tT/niter<<" msecs, #iters: "<<niter<<", kernel: "<<h<<", avg #points: "<<tPoints/niter<<endl;
}






void CPUBundling::computeDensity(float rad)													//Compute edge density on the GPU
{
	const float height = 1;																	//REMARK: The setting of the splat height is far less critical for this code than for
																							//the original KDEEB, since we have now a truly non-normalized floating-point density map
																							//(of which, we only use the normalized gradient)

	const int KERNEL_RADIUS = int(rad/2)*2;
	const int KERNEL_LENGTH = 2*KERNEL_RADIUS+1;											//Kernel #samples should always be odd

	for(unsigned int i=0; i<KERNEL_LENGTH; ++i)												//Create 1D filter kernel on CPU
	{
		float x = float(i)/float(KERNEL_LENGTH-1);
		x = fabs(x-0.5)/0.5;
		h_Kernel[i] = 1-x*x;																//parabolic filter
	}

	setConvolutionKernel(h_Kernel,KERNEL_LENGTH,KERNEL_RADIUS);								//Pass kernel to GPU

	initializeSiteLocations(density_estimation,d_densityRead,d_densityWrite,d_siteCount,d_points,numCtrlPts,height,fboSize,fboSize);
																							//Initialize per-pixel density on GPU from resident sites on GPU

	convolutionGPU(d_densityWrite,d_densityRead,fboSize,fboSize);							//Convolve sites on GPU with kernel to compute density field
}




void CPUBundling::computeDirections()														//Compute edge site-map on the GPU
{
	initializeSiteMap(d_siteMap,d_points,numCtrlPts,fboSize,fboSize);						//Initialize GPU site-map from resident sites on GPU
}







void CPUBundling::initEdgeProfile(EDGE_PROFILE ep)											//Create the 'edge profile' - the function that controls how strongly
{																							//an edge is bundled as we travel along it
	edge_profile = ep;

	for(int i=0;i<EDGE_PROFILE_SIZE;++i)													//We store this function in a sampled array, for ease of use later
	{
		float x = fabs(i-EDGE_PROFILE_SIZE/2.0)/(EDGE_PROFILE_SIZE/2.0);					//Distance to edge-midpoint (for making symmetric profiles)
		float p;
		switch(ep)
		{
		case PROFILE_UNIFORM:
			p = 1; break;
		case PROFILE_HOURGLASS:
			p = (x>0.7)? pow((1-x)/0.3,4):1; break;
		}

		h_edgeProfile[i] = p;
	}

	if (block_endpoints)
	{  h_edgeProfile[0] = h_edgeProfile[EDGE_PROFILE_SIZE-1] = 0;  }						//If we block endpoints, make sure the edge-profile is 0 there

	cudaMemcpy(d_edgeProfile,h_edgeProfile,EDGE_PROFILE_SIZE*sizeof(float),cudaMemcpyHostToDevice);		//Pass the edge-bundling profile to CUDA as well
}





void CPUBundling::applySmooth(const Polyline& pl,Polyline& npl,int j,int L,float t)
{																			//Helper function: smooths j-th point of 'pl', with a kernel of 'L' points,
   Point2d pc;																//and a smoothing-strength 't'. Result is put into 'npl'.
   int pcount = 0;
   int NP = pl.size();

   for(int k=j-L;k<=j+L;++k)
   {
	  if (k<0 || k>=NP) continue;
	  pc += pl[k];
	  ++pcount;
   }

   if (!pcount) return;
   Point2d np = pl[j]*(1-t) + pc*(t/pcount);
   npl[j] = np;
}



void CPUBundling::smoothCPU(float t,bool end_segs)							//Laplacian smoothing of graph-drawing edges
{
	int L;																	//Half-size of smoothing kernel width (in #points)
	if (end_segs)
		L = h/spl;
	else
		L = 4;

	for(int iter=0;iter<liter;++iter)										//Iterate smoothing:
	{
		int K=0;
		for(int i=0,NL=drawing->numNodes();i<NL;++i)						//For all nodes:
		{
			GraphDrawing::Row& row = (*drawing)(i);							//For all edges of a node:
			for(GraphDrawing::Row::iterator it=row.begin(),ie=row.end();it!=ie;++it)
			{
				Polyline& pl  = *it->second;
				Polyline  npl = pl;							//Don't smooth in-place

				float d_max_start,d_max_end;
				if (end_segs)
				{
					d_max_start = ms_diststart[K]+h;						//Get distances from start to 1st MS-point and
					d_max_end   = ms_distend[K++]+h;						//end to 2nd MS-point, both increased by kernel-size
				}

				int NP      = pl.size();
				int NP_half = NP/2;

				float d_start = 0;
				for(int j=1;j<=NP_half;++j)									//Smooth current edge - except its 1st endpoint (up to half edge):
				{
				   if (end_segs)											//Smooth points created by the CPU resampling, up to a kernel-length L _after_ the MS control point
				   {
					   d_start += pl[j].dist(pl[j-1]);
					   if (d_start > d_max_start) continue;					//Stop when we pass 1st MS-point with h
				   }

				   applySmooth(pl,npl,j,L,t);
				}

				float d_end = 0;
				for(int j=pl.size()-2;j>NP_half;--j)						//Smooth current edge - except its last endpoint (up to half edge):
				{
				   if (end_segs)											//Smooth points created by the CPU resampling, up to a kernel-length L _after_ the MS control point
				   {
					   d_end += pl[j].dist(pl[j+1]);
					   if (d_end > d_max_end) continue;						//Stop when we pass 2nd MS-point with h
				   }

				   applySmooth(pl,npl,j,L,t);
				}

				pl = npl;													//Edge was smoothed, done
			}
		}
	}
}
