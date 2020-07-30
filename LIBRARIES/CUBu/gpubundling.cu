#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "include/cudawrapper.h"
#include <curand_kernel.h>
#include "include/gpubundling.h"


texture<float,cudaTextureType2D,cudaReadModeElementType>		 texDensity;						//State var:  the float 2D density texture
texture<unsigned int,cudaTextureType1D,cudaReadModeElementType>	 texCount;							//!!
texture<BundlingPoint,cudaTextureType1D,cudaReadModeElementType> texSites;							//State var:  the float2 1D sites vector
texture<int,cudaTextureType1D,cudaReadModeElementType>			 texStarts;							//State var:
texture<int,cudaTextureType1D,cudaReadModeElementType>			 texEdges;							//State var:  start-offsets of all edges in texSites[]
texture<float,cudaTextureType1D,cudaReadModeElementType>		 texEdgeProfile;					//State var:  the edge profile (controlling the advection along an edge)
texture<unsigned int,cudaTextureType2D,cudaReadModeElementType>	 texInt2D;							//!!



__constant__ float												 c_Kernel[MAX_KERNEL_LENGTH];		//State var:  the kernel data (stored as constant for speed)
__constant__ int kernel_radius;																		//State var:  the current kernel radius
__constant__ int imageW,imageH;																		//State vars: sizes of the image used allover through the code
__device__   int numControlPoints;




//--- GPU-specific defines -----------------------------------------------------------------

#define IMAD(a, b, c) ( __mul24((a), (b)) + (c) )			//Maps to a single instruction on G8x / G9x / G10x

inline int iDivUp(int a, int b)								//Round a / b to nearest higher integer value
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

inline int iAlignUp(int a, int b)							//Align a to nearest higher multiple of b
{
    return (a % b != 0) ?  (a - a % b + b) : a;
}



__global__ void convolutionRowsKernel(float* output)
{
    const   int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const   int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
    if (ix >= imageW || iy >= imageH) return;							//Careful not to index outside image
																		//REMARK: For advection, we don't actually need the density field over the entire image,
    const float  x = (float)ix + 0.5f;									//		  but only over a 2x2 nbhood of the sampling points (for gradient computation).
    const float  y = (float)iy + 0.5f;									//		  However, for shading, we need it over all pixels covered by edges. Since it's
    float sum = 0;														//		  hard to limit computation there, we compute density over the entire image.

    /*
    bool close2site = false;
    for(short X = x-1; X <= x+1 && !close2site; ++X)
    for(short Y = y-1; Y <= y+1 && !close2site; ++Y)
    {
    if (tex2D(texDensity,X,Y))
    close2site = true;
    }

    if (close2site)
*/
	for(short k = -kernel_radius; k <= kernel_radius; ++k)
	sum += tex2D(texDensity,x+k,y)*c_Kernel[kernel_radius-k];

    output[IMAD(iy,imageW,ix)] = sum;
}


__global__ void convolutionColumnsKernel(float* output)
{
    const   int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const   int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
    if (ix >= imageW || iy >= imageH) return;							//Careful not to index outside image

    const float  x = (float)ix + 0.5f;
    const float  y = (float)iy + 0.5f;
    float sum = 0;

/*
    bool close2site = false;
    for(short X = x-1; X <= x+1 && !close2site; ++X)
    for(short Y = y-1; Y <= y+1 && !close2site; ++Y)
    {
        if (tex2D(texDensity,X,Y))
           close2site = true;
    }

    if (close2site)
*/
	for(short k = -kernel_radius; k <= kernel_radius; ++k)
		sum += tex2D(texDensity,x,y+k)*c_Kernel[kernel_radius-k];

	output[IMAD(iy,imageW,ix)] = sum;
}

void convolutionRowsColumns(float* output,cudaArray *a_Src,int imageW,int imageH,bool rows_or_columns)
{
    dim3 threads(NTHREADS_X,NTHREADS_Y);
    dim3 blocks(iDivUp(imageW,threads.x),iDivUp(imageH,threads.y));

    cudaBindTextureToArray(texDensity,a_Src);
	if (rows_or_columns)
		convolutionRowsKernel<<<blocks,threads>>>(output);
	else
		convolutionColumnsKernel<<<blocks,threads>>>(output);
	cudaUnbindTexture(texDensity);

    cudaDeviceSynchronize();
}


extern "C" void convolutionGPU(float* d_Output,cudaArray* a_Src,int imageW,int imageH)
{
	//1. Convolve on rows (a_Src -> d_Output)
	cudaDeviceSynchronize();

	convolutionRowsColumns(d_Output,a_Src,imageW,imageH,true);

    //2. Copy row-convolution to texture (d_Output -> a_Src)
    //   While CUDA kernels can't write to textures directly, this copy is inevitable
	cudaMemcpyToArray(a_Src,0,0,d_Output,imageW*imageH*sizeof(float),cudaMemcpyDeviceToDevice);
	cudaDeviceSynchronize();

    //3. Convolve on columns (a_Src -> d_Output)
	convolutionRowsColumns(d_Output,a_Src,imageW,imageH,false);
}





__global__ void kernelSiteInitCount(unsigned int* output,int numpts)		//Init the image output[] with the #sites at ech pixel
{
    int offs = IMAD(blockIdx.x, blockDim.x, threadIdx.x);

	if (offs < numpts)														//careful not to index outside the site-vector..
	{
	  const BundlingPoint p = tex1Dfetch(texSites,offs);					//find coords of current site
	  if (p.x<0) return;													//this is a marker: nothing to do

	  int site_offs = IMAD(int(p.y),imageW,int(p.x));						//Increment site-count for pixel under it
	  atomicAdd(output+site_offs,1);										//REMARK: This seems expensive, but it only occurs when sites DO overlap...
	}
}

__global__ void kernelSiteInitFloat(float* output,int numpts,float value)	//Init the image output[] with 'value' at all site locations
{
	int offs = IMAD(blockIdx.x,blockDim.x,threadIdx.x);

	if (offs < numpts)
	{
	  const BundlingPoint p  = tex1Dfetch(texSites,offs);					//find coords of current site
	  if (p.x<0) return;													//this is a marker: nothing to do

	  int site_offs = IMAD(int(p.y),imageW,int(p.x));
output[site_offs] += p.w; //!!!value;											//Set pixel under site to 'value'
	}																		//WARNING: This underestimates density where multiple sites fall on the same pixel
																			//due to the fact that += is not atomic on threads that want to write to the same pixel
}

__global__ void kernelSiteInitCount2Float(float* output,int imageW)			//Copy count image (from texCount) to floating-point image output[]
{
    const   int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const   int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
    if (ix >= imageW || iy >= imageH) return;								//Careful not to index outside image

	const int offs = IMAD(iy,imageW,ix);
    output[offs] = tex1Dfetch(texCount,offs);
}







extern "C" void initializeSiteLocations(CPUBundling::DENSITY_ESTIM dens_estim,cudaArray* a_Src,float* d_Output,unsigned int* d_Count,BundlingPoint* d_points,int npoints,float value,int imageW,int imageH)
{
    cudaMemcpyToSymbol(::imageW, &imageW, sizeof(int));					//Store these state-vars in const memory, for fast access later
    cudaMemcpyToSymbol(::imageH, &imageH, sizeof(int));

	int threads  = NTHREADS;											//Prepare the site-init kernel: this reads a vector of 2D sites
	int numpts_b = iAlignUp(npoints,threads);							//Find higher multiple of blocksize than # sites
	int blocks   = numpts_b/threads;

	cudaBindTexture(0,texSites,d_points);								//Bind 2D sites to a 1D texture

	if (dens_estim==CPUBundling::DENSITY_FAST)							//Fast density estimation: simply record if there's no or one..more sites/pixel
	{
		cudaMemset(d_Output,0,sizeof(float)*imageW*imageH);				 //Zero the density texture
		kernelSiteInitFloat<<<blocks,threads>>>(d_Output,npoints,value); //Set all site-locations in the density map to 'value'
	}
	else	//DENSITY_EXACT												//Exact density estimation, pass 1
	{																	//Count #sites/pixel (slower, uses atomic ops)
		cudaMemset(d_Count,0,sizeof(unsigned int)*imageW*imageH);
		kernelSiteInitCount<<<blocks,threads>>>(d_Count,npoints);		//Count #sites/pixel in d_Count
	}

	cudaDeviceSynchronize();
	cudaUnbindTexture(texSites);										//Done with the sites

	if (dens_estim==CPUBundling::DENSITY_EXACT)							//Exact density estimation, pass 2
	{
		dim3 threads(NTHREADS_X,NTHREADS_Y);
		dim3 blocks(iDivUp(imageW,threads.x),iDivUp(imageH,threads.y));
		cudaBindTexture(0,texCount,d_Count);
		kernelSiteInitCount2Float<<<blocks,threads>>>(d_Output,imageW);	//Simply copy d_Count to the floating-point d_Output
		cudaDeviceSynchronize();
		cudaUnbindTexture(texCount);
	}

	cudaMemcpyToArray(a_Src,0,0,d_Output,imageW*imageH*sizeof(float),cudaMemcpyDeviceToDevice);


    texDensity.addressMode[0] = cudaAddressModeClamp;                   //Not sure if this is needed, but I want to make sure that
    texDensity.addressMode[1] = cudaAddressModeClamp;                   //texDensity is bilinearly interpolated when read at floating-point locations
    texDensity.filterMode     = cudaFilterModeLinear;
    texDensity.normalized     = false;
}






__global__ void kernelSiteInitDirs(unsigned int* output, int numpts)		//Init the image output[] with the dirs at ech pixel
{
    int offs = IMAD(blockIdx.x, blockDim.x, threadIdx.x);

	if (offs < numpts)														//careful not to index outside the site-vector..
	{
	  BundlingPoint p = tex1Dfetch(texSites,offs);							//find current site
	  if (p.x<0) return;													//this is a marker: don't write it into the texture

	  unsigned int site_offs = IMAD(imageW,int(p.y),int(p.x));				//Find pixel under site
	  output[site_offs] = offs;												//Write its point-ID into the texture
	}
}



extern "C" void initializeSiteMap(unsigned int* siteMap,BundlingPoint* d_points,int npoints,int imageW,int imageH)
{
    cudaMemcpyToSymbol(::imageW, &imageW, sizeof(int));					//Store these state-vars in const memory, for fast access later
    cudaMemcpyToSymbol(::imageH, &imageH, sizeof(int));

	int threads  = NTHREADS;											//Prepare the site-init kernel: this reads a vector of 2D sites
	int numpts_b = iAlignUp(npoints,threads);							//Find higher multiple of blocksize than # sites
	int blocks   = numpts_b/threads;

	cudaBindTexture(0,texSites,d_points);								//Bind 2D sites to a 1D texture

																		//Set the site-IDs as values of their pixel locations in texture
	cudaMemset(siteMap,0,sizeof(unsigned int)*imageW*imageH);			//First, zero up the sitemap-texture
	kernelSiteInitDirs<<<blocks,threads>>>(siteMap,npoints);			//Next, write sites in texture
	cudaDeviceSynchronize();
	cudaUnbindTexture(texSites);										//Done with the site-texture
}



__device__ inline float2 edgeTangent(int offs)
{
	BundlingPoint p   = tex1Dfetch(texSites,offs);						//Get current point to advect
	BundlingPoint q   = tex1Dfetch(texSites,offs+1);					//Get current point to advect
	if (q.x<0)
	{
	   q = p;
	   p = tex1Dfetch(texSites,offs-1);
	}

	float eps = 1.0e-5;
	float2 tv = make_float2(q.x-p.x,q.y-p.y);
	float tvn = rsqrtf(tv.x*tv.x+tv.y*tv.y+eps);
	tv.x *= tvn;														//tv: tangent vector to edge at p
	tv.y *= tvn;

	return tv;
}




//REMARK: rho, in the dir-advection, cannot always be 1 - its length must depend on compatibility
//REMARK: we have 2 possibilities for dir-compatibility: initial dirs, or dirs of the bundled edges

//REMARK: The directional factor should be a function of
//		  -the length of the attracted edge: longer gets more dir-influence
//		  -the length of the attracting edge: ???

//REMARK: Modify ::draw (shading) to ignore shading if edge-dir (as given by tangent) is very different than grad density
//
//




__device__ float interpAngle(float a,float b,float blend)
{
	float ix = cosf(a);
	float iy = sinf(a);
	float jx = cosf(b);
	float jy = sinf(b);

	float x = (1-blend)*ix + blend*jx;
	float y = (1-blend)*iy + blend*jy;
	float n = rsqrtf(x*x+y*y);
	x *= n;
	y *= n;

	float res = atan2f(y,x);
	if (res<0) res += 2*M_PI;
	return res;
}


__device__ float dir2angle(const float2& d)
{
	float a = acosf(d.x);
	if (d.y<0) a += M_PI;
	return a;
}


__device__ float2 dirDensityGradient(int offs,int imageW,float rep_strength)
{																		//Compute density-gradient for advection that considers edge-directions
	const float        eps = 1.0e-4;
	const BundlingPoint  p = tex1Dfetch(texSites,offs);					//get current site, where we want to compute the dir. density
	float              a_p = p.w;										//get edge-direction at site 'p' (angle in [0,2*M_PI])

	float2 grad = make_float2(0,0);										//accumulates compatible tangent-vectors around p
	int minX  = fmaxf(0.0f,p.x-kernel_radius), maxX = fminf(float(imageW),p.x+kernel_radius);
	int minY  = fmaxf(0.0f,p.y-kernel_radius), maxY = fminf(float(imageW),p.y+kernel_radius);
	float r2    = kernel_radius*kernel_radius;

	int N=0;															//counts #sites that we gather from
	float comps = 0;
	for(int i=minX;i<=maxX;++i)											//scan circle of radius 'radius_kernel' around p:
	{																	//(this gathering step is SLOW)
	  int dx2 = (p.x-i)*(p.x-i);
	  for(int j=minY;j<=maxY;++j)
	  {
		  int d2 = dx2+(p.y-j)*(p.y-j);
		  if (d2>r2) continue;											//ignore points outside kernel_radius

		  unsigned int q = tex2D(texInt2D,i,j);							//get possible site at pixel (i,j)
		  if (q==0) continue;											//no site there? nothing to gather at p from (i,j)

		  float  a_q = tex1Dfetch(texSites,q).w;						//get edge-direction at site 'q' (angle in [0,2*M_PI])

		  float comp = fabsf(a_q-a_p);									//compute compatibility of site 'q' with site 'p'
		  if (comp>M_PI) comp = __fmaf_rz(2,M_PI,-comp);				//make comp reside in [0,pi]
		  comp = __fmaf_rz(-2/M_PI,comp,1);								//comp in [-1,1]: comp=1=parallel, comp=0=ortho, comp=-1=antiparallel
																		//hence: parallel edges attract themselves, antiparallel ones repel themselves

		  if (comp<0) comp *= rep_strength; 							//restrict repulsion

		  float2 g = make_float2(i-p.x,j-p.y);							//compute normalized density gradient due to site (i,j)
		  float gn = comp*__expf(-4*d2/r2);
		  g.x *= gn; g.y *= gn;

		  grad.x += g.x;												//accumulate gradient weighted by directional compatibility
		  grad.y += g.y;
		  comps += fabsf(comp);											//accumulate weights of summed gradients
		  ++N;
	  }
   }

   float r = grad.x*grad.x+grad.y*grad.y;								//normalize the resulting gradient
   if (r<1.0e-4) { grad.x = grad.y = 0; }
   else
   {
       float K = comps / N;
	   float rn = K*rsqrtf(r+eps);
	   grad.x *= rn; grad.y *= rn;
   }

   return grad;
}




__global__ void kernelAdvectSites(BundlingPoint* output,float h,float numpts)
{																			//Advect graph in density-gradient
    int offs = IMAD(blockIdx.x, blockDim.x, threadIdx.x);

	if (offs<numpts)
	{
		BundlingPoint p   = tex1Dfetch(texSites,offs);						//Get current point to advect

		if (p.x<0)															//Marker: copy it, don't advect it
		{
		    output[offs] = p;
		}
		else																//Regular point: advect it
		{
			float  v_d = tex2D(texDensity,p.x,p.y-1);						//Get density at that point and at its nbs
			float  v_l = tex2D(texDensity,p.x-1,p.y);
			float  v_r = tex2D(texDensity,p.x+1,p.y);
			float  v_t = tex2D(texDensity,p.x,p.y+1);
			BundlingPoint g = make_float4(v_r-v_l,v_t-v_d,p.z,p.w);			//Compute density gradient, simple forward difference method

			const float eps = 1.0e-4;										//Ensures we don't next get div by 0 for 0-length vectors
			float gn = g.x*g.x+g.y*g.y;
			if (gn<eps) gn = 0;
			else gn = rsqrtf(gn);											//Robustly normalize the gradient

			float  k = h*p.z*gn;                                            //k = displacement of current point 'p'
			g.x *= k; g.x += p.x;											//Advect current point
			g.y *= k; g.y += p.y;

      output[offs] = g;												//Write displaced point to 'output'

		}
	}
}









__global__ void kernelAdvectSitesDirectional(BundlingPoint* output,float h,float numpts,int imageW,float rep_strength)
{																			//Advect graph in density-gradient caused only by directionally-compatible sites
    int offs = IMAD(blockIdx.x, blockDim.x, threadIdx.x);

	if (offs<numpts)
	{
		BundlingPoint p   = tex1Dfetch(texSites,offs);						//Get current point to advect

		if (p.x<0)															//Marker: copy it, don't advect it
		{
		    output[offs] = p;
		}
		else																//Regular point: advect it
		{
			float2 grad = dirDensityGradient(offs,imageW,rep_strength);		//Compute directional density gradient at current point
			BundlingPoint g = make_float4(grad.x,grad.y,p.z,p.w);

			float  k = h*p.z;
			g.x *= k; g.x += p.x;											//Advect current point
			g.y *= k; g.y += p.y;

			output[offs] = g;												//Write displaced point to 'output'
		}
	}
}





extern "C" void advectSites(BundlingPoint* out_points,BundlingPoint* in_points,int npoints,float* a_Src,unsigned int* d_siteMap,int imageW,int imageH,
							float h,bool tangent,float rep_strength)
{																		//Advect the sites, one step, along its density gradient
	dim3 block = dim3(NTHREADS);										//Prepare the site-init kernel: this reads a vector of 2D sites
	int numpts_b = iAlignUp(npoints,block.x);							//Find higher multiple of blocksize than # sites
	dim3 grid  = dim3(numpts_b/block.x);								//Number of blocks, each of 'block' threads. Each thread advects a point.

	cudaBindTexture(0,texSites,in_points);								//Bind 2D sites to a 1D texture (for reading)

    if (!tangent)														//No directional bundling:
	{
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32,0,0,0,cudaChannelFormatKindFloat);
		cudaBindTexture2D(0,texDensity,a_Src,channelDesc,imageW,imageH,4*imageW);
																		//Bind density to a 2D texture (for reading)
		kernelAdvectSites<<<grid,block>>>(out_points,h,npoints);		//Advect the sites from in_points to out_points	in the density gradient
		cudaDeviceSynchronize();
		cudaUnbindTexture(texDensity);
	}
	else																//Directional bundling:
	{
		texInt2D.filterMode = cudaFilterModePoint;						//Bind side-ID image to a 2D texture (for reading)
		texInt2D.normalized = false;
		cudaChannelFormatDesc channelDesc2 = cudaCreateChannelDesc(32,0,0,0,cudaChannelFormatKindUnsigned);
		cudaBindTexture2D(0,texInt2D,d_siteMap,channelDesc2,imageW,imageH,4*imageW);

		kernelAdvectSitesDirectional<<<grid,block>>>(out_points,h,npoints,imageW,rep_strength);
		cudaDeviceSynchronize();										//Advect the sites from in_points to out_points	in the density gradient
		cudaUnbindTexture(texInt2D);
	}

	cudaUnbindTexture(texSites);										//Done with the textures
}



__global__ void kernelSmoothLines(BundlingPoint* out_points,int npoints,float t,int L)
{
    int offs = IMAD(blockIdx.x, blockDim.x, threadIdx.x);

	if (offs>=npoints) return;												//Care not to index outside the site array

	out_points += offs;

	BundlingPoint crtp = tex1Dfetch(texSites,offs);							//Is current point an end-of-line marker?
	if (crtp.x<0)															//If so, copy it w/o smoothing
	{
	   *out_points = crtp;
	}
	else																	//Smooth current point w.r.t. a window of [-L,L] points centered at it:
	{
	   float2 pc = make_float2(0,0);
	   unsigned char pcount = 0;

	   int km = (offs>L)? offs-L : 0;										//Make sure we don't pass left to 0-th point..
	   #pragma unroll
	   for(int k=offs-1;k>=km;--k)											//Gather points 'upstream' until end-of-kernel or beginning-of-line:
	   {
		  BundlingPoint pinp = tex1Dfetch(texSites,k);
		  if (pinp.x<0) break;												//Stop at line-beginning

		  pc.x += pinp.x;
		  pc.y += pinp.y;
		  ++pcount;
	   }

	   #pragma unroll
	   for(int kM=offs+L;offs<=kM;++offs)									//Gather points 'downstream' until end-of-kernel or end-of-line:
	   {
		  BundlingPoint pinp = tex1Dfetch(texSites,offs);
		  if (pinp.x<0) break;												//Stop at line-end

		  pc.x += pinp.x;
		  pc.y += pinp.y;
		  ++pcount;
	   }

	   t *= crtp.z;

	   const float k = t/pcount;											//Linear interpolation between point and average of its neighbors
	   crtp.x *= 1-t;
	   crtp.x += pc.x*k;
	   crtp.y *= 1-t;
	   crtp.y += pc.y*k;
	   *out_points = crtp;
	}
}



extern "C" void smoothLines(BundlingPoint* out_points,BundlingPoint* in_points,int npoints,float t,float h,float filter_kernel,int niter)	//Laplacian smoothing of graph-drawing edges
{
	dim3 threads = dim3(NTHREADS);										//Prepare the smooth kernel: this reads a vector of 2D sites
	int numpts_b = iAlignUp(npoints,threads.x);							//Find npoints upper-rounded to a multiple of block.x
	dim3 blocks  = dim3(numpts_b/threads.x);							//Find #blocks fitting numpts_b


	const int L = int(filter_kernel/h);									//Compute 1D Laplacian filter size, in #points, which corresponds to 'filter_kernel' space units
	if (L==0) return;													//Don't do smoothing if filter-size is zero..

	BundlingPoint *out = out_points, *inp = in_points;
	for(int i=0;i<niter;++i)											//Perform several Laplacian iterations:
	{
		cudaBindTexture(0,texSites,inp);								//Bind 2D sites to a 1D texture (for reading), unbinds any possibly-bound texture
		kernelSmoothLines<<<blocks,threads>>>(out,npoints,t,L);
		cudaDeviceSynchronize();

		BundlingPoint* tmp = out;										//Swap input vs output for next iteration
		out = inp;
		inp = tmp;
	}
	cudaUnbindTexture(texSites);										//Done with the texture
}











extern "C" void setConvolutionKernel(float* h_Kernel,int sz,int rad)
{
    cudaMemcpyToSymbol(c_Kernel, h_Kernel, sz * sizeof(float));
    cudaMemcpyToSymbol(kernel_radius, &rad, sizeof(int));
}


__global__ void kernelComputeShading(float* output,int imageW,int imageH,float lx,float ly,float lz,float h_max)
{																			//Compute shading from density gradient
    const   int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const   int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
    if (ix >= imageW-1 || iy >= imageH-1) return;							//Careful not to index outside image

    const float  x = (float)ix + 0.5f;
    const float  y = (float)iy + 0.5f;

	const float K = imageW/h_max;

	float v  = tex2D(texDensity,x,y);										//Compute normal to density graph from density gradient
	float vx = tex2D(texDensity,x+1,y);										//Normal to surface is the vector (drho/dx,drho/dy,-1)
	float vy = tex2D(texDensity,x,y+1);
	float gx = (vx-v)*K;
	float gy = (vy-v)*K;
	float g2 = gx*gx+gy*gy+1;
	float gn = rsqrtf(g2);													//Normalize the surface-normal

	int offs = IMAD(iy,imageW,ix);

	float shade = (gx*gn*lx+gy*gn*ly-gn*lz);								//Shading = dot-product of normal with light
	if (shade<0) shade=0;
    output[offs] = shade;
}





__global__ void kernelComputeShadingNorm(float* output,int imageW,int imageH,float lx,float ly,float lz,float R)
{																			//Compute shading from gradient of locally-normalized density
    const   int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const   int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
    if (ix >= imageW-1 || iy >= imageH-1) return;							//Careful not to index outside image

	float v_max = 0;
	const float R2 = R*R;
	for(int j=iy-R;j<=iy+R;++j)												//Determine the local density max in a
	for(int i=ix-R;i<=ix+R;++i)												//radius R around current point
	{																		//We'll use this to locally normalize the density in [0,1]
		float r2 = (i-ix)*(i-ix)+(j-iy)*(j-iy);								//This constrains the local gradient to decent ranges
		if (r2>R2) continue;
		float v  = tex2D(texDensity,i+0.5f,j+0.5f);							//(so we can next compute a nice gradient and thus shading)
		if (v>v_max) v_max = v;
	}

    if (v_max==0) v_max = 1;                                                //Careful not to divide by zero.

	v_max /= R;																//We normalize the density in [0,1]. The range of x and y is (imageW,imageH).
																			//To get a proportionate shape, whose normal is useful for shading, we
																			//scale the density in [0,R], corresponding to the height of a 'tube'
																			//of radius R. See also below how v_max is used.
    const float x = ix + 0.5f;
    const float y = iy + 0.5f;
	float v  = tex2D(texDensity,x,y);										//Compute normal to density graph from density gradient

	float vx = tex2D(texDensity,x+1,y);										//Normal to surface is the vector (drho/dx,drho/dy,-1)
	float vy = tex2D(texDensity,x,y+1);
	float gx = (vx-v)/v_max;												//gx,gy = partial derivatives of density w.r.t. x,y
	float gy = (vy-v)/v_max;
	float gl = gx*gx+gy*gy+1;
	float gn = rsqrtf(gl);													//Normalize the surface-normal

	float shade = (gx*gn*lx+gy*gn*ly-gn*lz);								//Shading = dot-product of normal with light
	if (shade<0) shade=0;

	int offs = IMAD(iy,imageW,ix);
    output[offs] = shade;
}



__global__ void kernelComputeShadingTube(float* output,int imageW,int imageH,float R)
{																			//Compute pseudo-shading from density. Take care, this is NOT a physically correct
                                                                            //shading. It is simply the locally normalized density, interpreted as luminance.

//WARNING:  This function is not fully correct. The issue is that R varies continuously (from the caller), since it's a float.
//          But the sampling of texDensity is done at increments of +/- 1 pixel. So, small changes of R may not be 'felt' by this code
//          when computing v_max (of texDensity) within a kernel of radius R. As such, when smoothly varying R in the caller,
//          the shading computed by this (output[]) will exhibit some sharp jumps when R goes over integer bounds (like from 3.9 to 4.1).
//


    const   int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const   int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
    if (ix >= imageW-1 || iy >= imageH-1) return;							//Careful not to index outside image

	float v_max = 0;
	const float R2 = R*R;
	for(int j=iy-R;j<=iy+R;++j)												//Determine the local density max in a
	for(int i=ix-R;i<=ix+R;++i)												//radius R around current point
	{																		//We'll use this to locally normalize the density in [0,1]
		float r2 = (i-ix)*(i-ix)+(j-iy)*(j-iy);
		if (r2>R2) continue;
		float v  = tex2D(texDensity,i+0.5f,j+0.5f);
		if (v>v_max) v_max = v;
	}

    if (v_max==0) v_max=1;                                                  //Avoid division by zero

	float v  = tex2D(texDensity,ix,iy);                                     //Compute density locally normalized in [0,1]
	int offs = IMAD(iy,imageW,ix);
    output[offs] = v/v_max;                                                 //This is the normalized density at the current point.
                                                                            //More pronounced tube effects can be gotten by raising this to pow(2)
}





__global__ void kernelComputeDT(float* output,int imageW,int imageH)		//Compute shading from density gradient
{
    const   int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const   int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
    if (ix >= imageW-1 || iy >= imageH-1) return;							//Careful not to index outside image

	const float v_min = 0.1;

	const int R = 10;
	int d2min = R*R;
	for(int i=ix-R;i<=ix+R;++i)
	for(int j=iy-R;j<=iy+R;++j)
	{
		const float  x = (float)i + 0.5f;
		const float  y = (float)j + 0.5f;
		float v  = tex2D(texDensity,x,y);										//Compute normal to density graph from density gradient
		if (v<v_min)
		{
			float d2 = (i-ix)*(i-ix)+(j-iy)*(j-iy);
			if (d2<d2min) d2min=d2;
		}
	}

	int offs = IMAD(iy,imageW,ix);
    output[offs] = sqrtf(d2min);
}



extern "C" void computeDT(float* d_Output,cudaArray* a_Src,int imageW,int imageH)
{
    dim3 threads(NTHREADS_X,NTHREADS_Y);
    dim3 blocks(iDivUp(imageW,threads.x),iDivUp(imageH,threads.y));

	cudaMemcpyToArray(a_Src,0,0,d_Output,imageW*imageH*sizeof(float),cudaMemcpyDeviceToDevice);		//Copy density to a_Src, since we'll overwrite it

	cudaBindTextureToArray(texDensity,a_Src);														//Bind density to a 2D texture (for reading)

	kernelComputeDT<<<blocks,threads>>>(d_Output,imageW,imageH);									//Compute DT from density
	cudaDeviceSynchronize();

	cudaUnbindTexture(texDensity);
}





extern "C" void computeShading(float* d_Output,cudaArray* a_Src,int imageW,int imageH,const Point3d& light,float radius,bool tube_style)
{
    dim3 threads(NTHREADS_X,NTHREADS_Y);
    dim3 blocks(iDivUp(imageW,threads.x),iDivUp(imageH,threads.y));

	cudaMemcpyToArray(a_Src,0,0,d_Output,imageW*imageH*sizeof(float),cudaMemcpyDeviceToDevice);			//Copy density to a_Src, since we'll overwrite it


	cudaBindTextureToArray(texDensity,a_Src);															//Bind density to a 2D texture (for reading)

	if (tube_style)
	    kernelComputeShadingTube<<<blocks,threads>>>(d_Output,imageW,imageH,radius);						  //Compute tube-like shading
    else
	    kernelComputeShadingNorm<<<blocks,threads>>>(d_Output,imageW,imageH,light.x,light.y,light.z,radius);  //Compute shading from density gradient
	cudaDeviceSynchronize();

	cudaUnbindTexture(texDensity);
}



//------------------------------


__device__ inline float dist(const BundlingPoint& p,const BundlingPoint& q)
{
	return sqrtf((p.x-q.x)*(p.x-q.x)+(p.y-q.y)*(p.y-q.y));
}



__global__ void kernelResampleCount(int* e_count,int n_edges,float delta)
{
        int e_idx = IMAD(blockIdx.x, blockDim.x, threadIdx.x);			//Get edge number
	if (e_idx>=n_edges) return;						//Careful not to index outside edge-vector


	int	        i    = tex1Dfetch(texStarts,e_idx);			//Find 1st edge-point in texSites
	BundlingPoint   prev = tex1Dfetch(texSites,i);				//First point on current edge
	float		crtd = delta;
	int	        n_newp = 1;						//Add 1st point of input polyline to resampled one
	++i;


	BundlingPoint crt = tex1Dfetch(texSites,i);
	for(;;)									//resample input polyline:
	{
		float newdist = dist(crt,prev);					//distance from last resampled point to i-th input point

		if (newdist<crtd)						//i-th input point closer to 'prev' than remaining fraction of delta:
		{								//skip i-th point
			crtd -= newdist;
			prev  = crt;
			++i;
			crt = tex1Dfetch(texSites,i);
			if (crt.x<0) break;
		}
		else								//i-th input point farther from 'prev' than remaining fraction of delta:
		{
			float t = crtd/newdist;
			prev.x = prev.x*(1-t) + crt.x*t;			//add new point to resampling
			prev.y = prev.y*(1-t) + crt.y*t;			//add new point to resampling
			++n_newp;
			crtd = delta;						//reset delta to whatever we want to achieve
		}
	}

	if (crtd<delta) ++n_newp;
	++n_newp;								//include marker in #points to be stored for this edge
  if(n_newp == 2) ++n_newp;

	e_count[e_idx] = n_newp;						//save #points/edge to generate after resampling
}



__global__ void kernelResample(BundlingPoint* new_pts,int n_edges,float delta,float jitter,curandState* state)
{

        int e_idx = IMAD(blockIdx.x, blockDim.x, threadIdx.x);			//Get edge number
	if (e_idx>=n_edges) return;						//Careful not to index outside edge-vector


	int			i    = tex1Dfetch(texStarts,e_idx);		//Find 1st edge-point in texSites
	BundlingPoint           prev = tex1Dfetch(texSites,i);			//First point on current edge

	float		        crtd = delta;
	int			n_newp = tex1Dfetch(texEdges,e_idx);
	int			inxt = tex1Dfetch(texEdges,e_idx+1);
	int			NP   = inxt-n_newp-2;
	if (NP<1)		NP=1;

	const int		EPSZ = EDGE_PROFILE_SIZE-1;

	int omp = 0;

	new_pts += n_newp;							//Here will the resampled edge's points be placed

	*new_pts++ = prev;							//add 1st point of input polyline to resampled one
	++i;

	++omp;

	curandState lState = state[threadIdx.x];				//cache random generator for speed, since we'll modify it locally

	BundlingPoint crt = tex1Dfetch(texSites,i);
	for(int j=1;;)								//resample input polyline:
	{
		float newdist = dist(crt,prev);					//distance from last resampled point to i-th input point

		if (newdist<crtd)						//i-th input point closer to 'prev' than remaining fraction of delta:
		{								//skip i-th point
			crtd -= newdist;
			prev  = crt;
			++i;
			crt = tex1Dfetch(texSites,i);
			if (crt.x<0) break;
		}
		else								//i-th input point farther from 'prev' than remaining fraction of delta:
		{
			float t  = crtd/newdist;
			float r  = curand_uniform(&lState)*2-1;			//generate random number in [-1..1]
			float rt = t*(1+r*jitter);				//jitter currently-generated point
			BundlingPoint rp;
			rp.x   = prev.x*(1-rt) + crt.x*rt;
			rp.y   = prev.y*(1-rt) + crt.y*rt;

			int pidx = int(j*EPSZ/NP);
			rp.z = tex1Dfetch(texEdgeProfile,pidx);			//apply edge profile on newly, resampled, edge
			rp.w = crt.w; //!!!interpAngle(prev.w,crt.w,rt);

			*new_pts++ = rp;					//add new resampled point to output

			++omp;

			if (omp==NP) break;

			prev.x = prev.x*(1-t) + crt.x*t;			//keep NON-jittered point as next point;
			prev.y = prev.y*(1-t) + crt.y*t;			//this ensures we get here EXACTLY the same polyline sampling as in kernelResampleCount()

			crtd = delta;						//reset delta to whatever we want to achieve
			++j;

		}
	}

  // For fetching the last point of the last edge, the texStarts array is not
  // filled correctly, so we just iterate over its points to fetch the endpoint
  // correctly
  if(e_idx==n_edges-1){
    BundlingPoint current_point, last;
    int st = tex1Dfetch(texStarts, e_idx);
    while(1) {
      current_point = tex1Dfetch(texSites, ++st);
      if(current_point.x<0) break;
      last = current_point;
    }
	  *new_pts = last;
  }else{
	  *new_pts = tex1Dfetch(texSites,  tex1Dfetch(texStarts,e_idx+1)-2);
  }

	++omp;
  *new_pts++;

	if (omp<NP+1)
	{
	   *new_pts = *(new_pts-1);
	   ++new_pts;
 	   ++omp;
	}


	//!!if (crtd<delta)
	  //!!*new_pts++ = tex1Dfetch(texSites,i-1);

	if (omp!=NP+1)
		printf("***** Expected: %d, generated %d\n",NP+1,omp);

	new_pts->x = -1;							//add end-of-line marker

	state[threadIdx.x] = lState;						//update random generator
}






__global__ void kernelOffs(int* edges,int n_edges)
{
	//if (blockIdx.x==0 && threadIdx.x==0)
	{
		int cprev = edges[0];											//2. From 1, compute start-offset of resampled edges. Knowing this allows us to parallelize
		edges[0] = 0;													//   the resampling and writing the resampled points (in pass 3 below)
		for(int i=1;i<n_edges;++i)
		{
			int tmp = edges[i];
			edges[i] = edges[i-1] + cprev;
			cprev = tmp;
		}

		numControlPoints = edges[n_edges-1]+cprev;
	}
}



__global__ void kernelRandomInit(curandState* state)					//Initialize one random number generator state
{
	int id = threadIdx.x;
	curand_init(1234,id,0,state+id);
}

extern "C" void random_init(curandState* d_states)						//Initialize NTHREADS random generators.
{																		//We'll use them later in kernels when we need random numbers.
	kernelRandomInit<<<1,NTHREADS>>>(d_states);
	cudaDeviceSynchronize();
}


extern "C" void resample(BundlingPoint* new_points,int& n_outpoints,int* out_edges,BundlingPoint* in_points,int n_inpoints,int* in_edges,int* h_edges,int n_edges,float delta,curandState* d_states,float jitter,
						 float* d_edgeProfile)
{
	int threads    = int(NTHREADS);										//Prepare the resample kernel
	int numedges_b = iAlignUp(n_edges,threads);							//Find higher multiple of blocksize than # edges
	int blocks     = int(numedges_b/threads);

	cudaBindTexture(0,texSites,in_points);								//Bind 2D sites to a 1D texture
	cudaBindTexture(0,texStarts,in_edges);								//Bind edge-start offsets in above vector to another 1D texture

	kernelResampleCount<<<blocks,threads>>>(out_edges,n_edges,delta);	//1. Compute #points that resampling produces on each edge. Store this in out_edges[]
	cudaDeviceSynchronize();

	cudaMemcpy(h_edges,out_edges,n_edges*sizeof(int),cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	int cprev = h_edges[0];												//2. From 1, compute start-offset of resampled edges. Knowing this allows us to parallelize
	h_edges[0] = 0;														//   the resampling and writing the resampled points (in pass 3 below)
	for(int i=1;i<n_edges;++i)											//REMARK: This is the only still required CPU-GPU communication for the bundling algorithm..
	{
		int tmp = h_edges[i];
		h_edges[i] = h_edges[i-1] + cprev;
		cprev = tmp;
	}
	int NP = h_edges[n_edges-1]+cprev;
	h_edges[n_edges] = NP;												//Put an extra item at end, equal to the #points

	cudaMemcpy(out_edges,h_edges,(n_edges+1)*sizeof(int),cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	cudaBindTexture(0,texEdges,out_edges);											//3. Resample edges, producing same #points/edge as in pass 1. Put results in
	cudaBindTexture(0,texEdgeProfile,d_edgeProfile);								//   contiguous vector 'new_points'
	kernelResample<<<blocks,threads>>>(new_points,n_edges,delta,jitter,d_states);
	cudaDeviceSynchronize();

	cudaUnbindTexture(texEdgeProfile);
	cudaUnbindTexture(texEdges);
	cudaUnbindTexture(texSites);
	cudaUnbindTexture(texStarts);

	n_outpoints = NP;
}





__global__ void kernelDrawing2GL(GLBundlingPoint* out_points,int n_edges)
{
    int e_idx = IMAD(blockIdx.x, blockDim.x, threadIdx.x);              //Get edge number in texStarts
    if (e_idx>=n_edges) return;                                         //Careful not to index outside edge-vector

    int	offs    = tex1Dfetch(texStarts,e_idx);                          //Find idx in texSites of 1st edge-point on current edge

    out_points += offs;                                                 //Find position of 1st edge-point in texSites (and out_points too)

    GLBundlingPoint prev_point;

    for(;;++offs)
    {
        BundlingPoint   crtp = tex1Dfetch(texSites,offs);               //Current point on edge
        if (crtp.x<0)                                                   //End-of-edge marker?
        {                                                               //Simply copy previous point (always exists)
            *out_points = prev_point;
            break;
        }
        else                                                            //Not end-of-edge marker?
        {
            prev_point.coord  =  make_float2(crtp.x,crtp.y);                   //Remember it for next iteration
            prev_point.rgba[0] = 255;
            prev_point.rgba[1] = 128;
            prev_point.rgba[2] = 0;
            prev_point.rgba[3] = 255;


            *out_points = prev_point;                                   //Copy it to output
        }

        ++out_points;
    }
}




extern "C" void drawing2GL(GLBundlingPoint* gl_points, BundlingPoint* in_points, int n_inpoints, int* in_edges, int n_edges)
{
    int threads    = int(NTHREADS);										//Prepare the resample kernel
    int numedges_b = iAlignUp(n_edges,threads);							//Find higher multiple of blocksize than # edges
    int blocks     = int(numedges_b/threads);

    cudaBindTexture(0,texSites,in_points);								//Bind 2D sites to a 1D texture
    cudaBindTexture(0,texStarts,in_edges);								//Bind edge-start offsets in above vector to another 1D texture

    kernelDrawing2GL<<<blocks,threads>>>(gl_points,n_edges);

    cudaDeviceSynchronize();

    cudaUnbindTexture(texStarts);
    cudaUnbindTexture(texSites);


//--------------------------------------------


}
