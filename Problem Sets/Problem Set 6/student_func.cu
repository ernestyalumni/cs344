//Udacity HW 6
//Poisson Blending

/* Background
   ==========

   The goal for this assignment is to take one image (the source) and
   paste it into another image (the destination) attempting to match the
   two images so that the pasting is non-obvious. This is
   known as a "seamless clone".

   The basic ideas are as follows:

   1) Figure out the interior and border of the source image
   2) Use the values of the border pixels in the destination image 
      as boundary conditions for solving a Poisson equation that tells
      us how to blend the images.
   
      No pixels from the destination except pixels on the border
      are used to compute the match.

   Solving the Poisson Equation
   ============================

   There are multiple ways to solve this equation - we choose an iterative
   method - specifically the Jacobi method. Iterative methods start with
   a guess of the solution and then iterate to try and improve the guess
   until it stops changing.  If the problem was well-suited for the method
   then it will stop and where it stops will be the solution.

   The Jacobi method is the simplest iterative method and converges slowly - 
   that is we need a lot of iterations to get to the answer, but it is the
   easiest method to write.

   Jacobi Iterations
   =================

   Our initial guess is going to be the source image itself.  This is a pretty
   good guess for what the blended image will look like and it means that
   we won't have to do as many iterations compared to if we had started far
   from the final solution.

   ImageGuess_prev (Floating point)
   ImageGuess_next (Floating point)

   DestinationImg
   SourceImg

   Follow these steps to implement one iteration:

   1) For every pixel p in the interior, compute two sums over the four neighboring pixels:
      Sum1: If the neighbor is in the interior then += ImageGuess_prev[neighbor]
             else if the neighbor in on the border then += DestinationImg[neighbor]

      Sum2: += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)

   2) Calculate the new pixel value:
      float newVal= (Sum1 + Sum2) / 4.f  <------ Notice that the result is FLOATING POINT
      ImageGuess_next[p] = min(255, max(0, newVal)); //clamp to [0, 255]


    In this assignment we will do 800 iterations.
   */

// EY : I added this to time stuff, GpuTimer
#include "timer.h"


#include "utils.h"
#include <thrust/host_vector.h>


#include <algorithm>

__device__ int bound_check(const int index, const int maxbound) {
	int index_val = index;
	index_val = min( max(index_val,0), maxbound-1);
	return index_val;
}

// flatten with bounds
__device__ int flatten_w_bound(int i_x, int i_y, int bound_x, int bound_y) {
	return bound_check( i_x, bound_x) + bound_x * bound_check( i_y, bound_y) ;
}

__global__ void create_mask(uchar4* d_sourceImg, unsigned char* d_mask, 
		const size_t srcSize) { 
	int k_x = threadIdx.x + blockDim.x * blockIdx.x;
	
	uchar4 tempval = d_sourceImg[k_x] ; 
	unsigned char mask_val = 
		((tempval.x + tempval.y + tempval.z) < 3 * 255 ) ? 1 : 0 ;  // return true or 1 if not white, return false or 0 if white
	d_mask[k_x] = mask_val;
}

__device__ unsigned char masked_sh(uchar4 val) {
	unsigned char mask_val = 
		((val.x + val.y + val.z)< 3 * 255) ? 1 : 0 ;
	return mask_val; }

// get 2d position from block
__device__
int2 get2dPos() { 
	return make_int2(
		blockIdx.x * blockDim.x + threadIdx.x,
       	blockIdx.y * blockDim.y + threadIdx.y
	);
}

// check whether a a value is within the image bounds
__device__
bool withinBounds(const int x, const int y, const size_t numRowsSource, const size_t numColsSource) {
    return ((x < numColsSource) && (y < numRowsSource));
}

__device__
bool masked(uchar4 val) {
	return (val.x != 255 || val.y != 255 || val.z != 255);
}

__device__
int getm(int x, int y, size_t numColsSource) {
	return y*numColsSource + x;
}

__global__
void maskPredicateKernel(
	const uchar4* const d_sourceImg,
	int* d_borderPredicate,
	int* d_interiorPredicate,
	const size_t numRowsSource,
	const size_t numColsSource) {
	
    const int2 p = get2dPos();
	const int  m = getm(p.x, p.y, numColsSource);
    
    if(!withinBounds(p.x, p.y, numRowsSource, numColsSource))
         return;

 	// run through each pixel and determine if its 
	// on the border, or if its on the interior border
	
	if(masked(d_sourceImg[m])) {
		int inbounds = 0;
		int interior = 0;

		// count how many of our neighbors are masked,
		// and how many neighbors we have
		if (withinBounds(p.x, p.y+1, numRowsSource, numColsSource)) {
			inbounds++;
			if(masked(d_sourceImg[getm(p.x, p.y+1, numColsSource)]))
				interior++;		
	
		}
		if (withinBounds(p.x, p.y-1, numRowsSource, numColsSource)) {
			inbounds++;
			if(masked(d_sourceImg[getm(p.x, p.y-1, numColsSource)]))
				interior++;		
	
		}
		if (withinBounds(p.x+1, p.y, numRowsSource, numColsSource)) {
			inbounds++;
			if(masked(d_sourceImg[getm(p.x+1, p.y, numColsSource)]))
				interior++;		
		}
		if (withinBounds(p.x-1, p.y, numRowsSource, numColsSource)) {
			inbounds++;
			if(masked(d_sourceImg[getm(p.x-1, p.y, numColsSource)]))
				interior++;		
		}

		// clear out the values so we don't
		// have to memset this destination stuff
		d_interiorPredicate[m] = 0;
		d_borderPredicate[m]   = 0;
	
		// if all our neighbors are masked, then its interior
		if(inbounds == interior) {
			d_interiorPredicate[m] = 1;
		} else if (interior > 0) {
			d_borderPredicate[m] = 1;
		}
	}
}

__global__ void maskPredicate_sh(uchar4* d_sourceImg, int* d_border, int* d_Int,const size_t numCols,const size_t numRows) {
	int k_x = threadIdx.x + blockIdx.x * blockDim.x ; 
	int k_y = threadIdx.y + blockIdx.y * blockDim.y ; 
	
	int k = flatten_w_bound( k_x,k_y, numCols, numRows);
	
	extern __shared__ uchar4* sh_in[];  // it'll be of size blockDim.x*blockDim.y (i.e. that's the number of elements)
	const int RAD = 1;
	const int S_x = ((int) blockDim.x + 2*RAD) ; 
	const int S_y = ((int) blockDim.y + 2*RAD) ; 

	const int s_x = threadIdx.x + RAD;
	const int s_y = threadIdx.y + RAD ;
	
	int l_x = 0;
	int l_y = 0;
	for (int ind_x = threadIdx.x; ind_x < S_x; ind_x += ((int) blockDim.x)) { 
		for (int ind_y = threadIdx.y; ind_y < S_y; ind_y += ((int) blockDim.y) ) { 
			l_x = bound_check( ind_x - RAD + ((int) blockIdx.x * blockDim.x) , numCols);
			l_y = bound_check( ind_y - RAD + ((int) blockIdx.y * blockDim.y) , numRows);
			
//			sh_in[flatten_w_bound(ind_x,ind_y,S_x,S_y)] = 
//				d_sourceImg[flatten_w_bound( l_x,l_y, numCols,numRows)]; 
		}
	}
	
	if ((k_x >= numCols) || (k_y >= numRows)) { return ; }
	__syncthreads();

//	if ((k_x >= numCols-1) || (k_y >= numRows-1) || (k_x < 1) || (k_y < 1)) { return ; }
//	__syncthreads();

/*
	if ( masked_sh( sh_in[ flatten_w_bound( s_x,s_y,S_x,S_y)]) ) {
		if ( ( masked_sh( sh_in[flatten_w_bound(s_x,s_y-1,S_x,S_y)]) == 1) && 
			 ( masked_sh( sh_in[flatten_w_bound(s_x,s_y+1,S_x,S_y)]) == 1) &&
			 ( masked_sh( sh_in[flatten_w_bound(s_x-1,s_y,S_x,S_y)]) == 1) &&
			 ( masked_sh( sh_in[flatten_w_bound(s_x+1,s_y,S_x,S_y)]) == 1) ) {
			d_interiorPredicate[ k ] = 1 ;
			d_borderPredicate[ k ] = 0; }
		else {
			d
	*/
//	if ( masked_sh( sh_in[ flatten_w_bound( s_x,s_y,S_x,S_y)]) ) {
	
	
//	}
	
}

__global__
void separateChannelsKernel(
	const uchar4* const inputImageRGBA,
	float* const redChannel,
	float* const greenChannel,
	float* const blueChannel,
	size_t numRows,
	size_t numCols)
{
    const int2 p = get2dPos();
	const int  m = getm(p.x, p.y, numCols);
    
    if(!withinBounds(p.x, p.y, numRows, numCols))
         return;

  	redChannel[m]   = (float)inputImageRGBA[m].x;
  	greenChannel[m] = (float)inputImageRGBA[m].y;
  	blueChannel[m]  = (float)inputImageRGBA[m].z;
}


// This kernel takes in an image represented as a char4 and splits
// it into three images consisting of only one color channel each
// note that I use a global memory, map, 1-to-1 (bijective) memory access 
// as we don't need to load into shared memory since repeated global memory access isn't needed
// also notice that I "coalesce" the warps by considering the arrays as 1-dimensional arrays, and not 
// 2-dimensional grids that were "strided"
__global__ void separateChannels(uchar4* d_sourceImg, 
							const size_t numColsSource, const size_t numRowsSource, 
							unsigned char* d_red_src, 
							unsigned char* d_blue_src,
							unsigned char* d_green_src) {
	
	int k = threadIdx.x + blockIdx.x * blockDim.x;
	int N = numColsSource * numRowsSource ; 
	if (k >= N) { return ;}
	
	uchar4 val = d_sourceImg[ k];
	
	d_red_src[k]   = val.x;
	d_blue_src[k]  = val.y;
	d_green_src[k] = val.z; 	

}


// copy computed values for the interior into the output
__global__ void finalcopyChannels(uchar4* d_blendedImg, 
							const size_t numCols, const size_t numRows, 
							float* const d_red, 
							float* const d_blue,
							float* const d_green) {
	
	int k = threadIdx.x + blockIdx.x * blockDim.x;

	int N = numCols * numRows ; 
	if (k >= N) { return ;}

	uchar4 val;
	val.x = ((unsigned char) d_red[k]);
	val.y = ((unsigned char) d_blue[k]);
	val.z = ((unsigned char) d_green[k]);
	d_blendedImg[k] = val;
}


// initialize (2) float buffers, from the unsigned char rbg channels
__global__ void init_fbuffers(const unsigned char* d_red_src,
								const unsigned char* d_blue_src,
								const unsigned char* d_green_src, 
			float * d_blendedValsRed_1, float * d_blendedValsRed_2,
			float * d_blendedValsBlue_1, float * d_blendedValsBlue_2,
			float * d_blendedValsGreen_1, float * d_blendedValsGreen_2, 
			const size_t numColsSource, const size_t numRowsSource) {
	
	int k = threadIdx.x + blockIdx.x * blockDim.x;
	int N = numColsSource*numRowsSource;
	if (k>=N) { return ;}
	
	float temp_red = ((float) d_red_src[k]);
	d_blendedValsRed_1[k] = temp_red;
	d_blendedValsRed_2[k] = temp_red;

	float temp_blue = ((float) d_blue_src[k]);
	d_blendedValsBlue_1[k] = temp_blue;
	d_blendedValsBlue_2[k] = temp_blue;

	float temp_green = ((float) d_green_src[k]);
	d_blendedValsGreen_1[k] = temp_green;
	d_blendedValsGreen_2[k] = temp_green;
	
}

__global__
void recombineChannelsKernel(
	uchar4* outputImageRGBA,
	float* const redChannel,
	float* const greenChannel,
	float* const blueChannel,
	size_t numRows,
	size_t numCols)
{
    const int2 p = get2dPos();
	const int  m = getm(p.x, p.y, numCols);
    
    if(!withinBounds(p.x, p.y, numRows, numCols))
         return;
	
	outputImageRGBA[m].x = (char)redChannel[m];
	outputImageRGBA[m].y = (char)greenChannel[m];
	outputImageRGBA[m].z = (char)blueChannel[m];
}

__global__
void jacobiKernel(
	float* d_in,
	float* d_out,
	const int* d_borderPredicate,
	const int* d_interiorPredicate,
	float* d_source,
	float* d_dest,
	size_t numRows,
	size_t numCols)
{
    const int2 p = get2dPos();
	const int  m = getm(p.x, p.y, numCols);
    
    if(!withinBounds(p.x, p.y, numRows, numCols))
         return;

	// calculate these values as indicated in the videos

	int lm;
	if(d_interiorPredicate[m]==1) {
		float a = 0.f, b=0.f, c=0.0f, d=0.f;
		float sourceVal = d_source[m];

		if(withinBounds(p.x, p.y+1, numRows, numCols)) {
			d++;
			lm = getm(p.x, p.y+1, numCols);
			if(d_interiorPredicate[lm]==1) {
				a += d_in[lm];
			} else if(d_borderPredicate[lm]==1) {
				b += d_dest[lm];
			}
			c += (sourceVal-d_source[lm]);
		}
		
		if(withinBounds(p.x, p.y-1, numRows, numCols)) {
			d++;
			lm = getm(p.x, p.y-1, numCols);
			if(d_interiorPredicate[lm]==1) {
				a += d_in[lm];
			} else if(d_borderPredicate[lm]==1) {
				b += d_dest[lm];
			}
			c += (sourceVal-d_source[lm]);
		}
		
		if(withinBounds(p.x+1, p.y, numRows, numCols)) {
			d++;
			lm = getm(p.x+1, p.y, numCols);
			if(d_interiorPredicate[lm]==1) {
				a += d_in[lm];
			} else if(d_borderPredicate[lm]==1) {
				b += d_dest[lm];
			}
			c += (sourceVal-d_source[lm]);
		}
		
		if(withinBounds(p.x-1, p.y, numRows, numCols)) {
			d++;
			lm = getm(p.x-1, p.y, numCols);
			if(d_interiorPredicate[lm]==1) {
				a += d_in[lm];
			} else if(d_borderPredicate[lm]==1) {
				b += d_dest[lm];
			}
			c += (sourceVal-d_source[lm]);
		}
		
		d_out[m] = min(255.f, max(0.0, (a + b + c)/d));
	} else {
		d_out[m] = d_dest[m];
	}
	
}

void your_blend(const uchar4* const h_sourceImg,  //IN
                const size_t numRowsSource, const size_t numColsSource,
                const uchar4* const h_destImg, //IN
                uchar4* const h_blendedImg) //OUT
{
	size_t srcSize = numRowsSource * numColsSource;
	
	////////////////////////////////////////////////////////////////////
	// grid, block dimensions
	////////////////////////////////////////////////////////////////////
	// M_mask, number of threads in a single thread block for 
	// create_mask, that creates a mask
	////////////////////////////////////////////////////////////////////
	const int M_mask = 1024;
	const int gridsize_mask = (srcSize + M_mask -1)/M_mask;

	////////////////////////////////////////////////////////////////////
	// M_IntBorder, number of threads in a single thread block for 
	// simple_computeIntBorder, that computes whether pixel is an interior point or border point
	// note that the way it is right now, 2 threads will be idle at the "ends"
	// but as Luebke taught us, the speed up/optimization gained in not launching these 2 threads is small
	dim3 M_IntBorder(32,32);
	dim3 gridsize_IntBorder( (numColsSource + M_IntBorder.x -1)/M_IntBorder.x, 
							(numRowsSource + M_IntBorder.y-1)/M_IntBorder.y );

	////////////////////////////////////////////////////////////////////
	// separateChannels
	const int M_separate = 1024;
	const int gridsize_separate = (srcSize+M_separate-1)/M_separate ;

	// init_fbuffers - initialize 2 float buffers
	const int M_init_buf = 1024;
	const int gridsize_init_buf = (srcSize+M_init_buf-1)/M_init_buf ;

	// computeG
	// remember to allocate shared memory, 3x
	dim3 M_G(128,8);
	dim3 gridsize_G( (numColsSource + M_G.x -1)/M_G.x , (numRowsSource + M_G.y-1)/M_G.y);

	// computeG
	// remember to allocate shared memory, 2x
	dim3 M_Iter(128,8);
	dim3 gridsize_Iter( ( numColsSource + M_Iter.x-1)/M_Iter.x, (numRowsSource + M_Iter.y - 1)/M_Iter.y);

	// copy swap
	const int M_swap = 1024;
	const int gridsize_swap = (srcSize+M_swap-1)/M_swap ;

	////////////////////////////////////////////////////////////////////
	// END of grid, block dimensions
	////////////////////////////////////////////////////////////////////

	
	// first push the dest and source onto the gpu
	size_t imageSize = numRowsSource*numColsSource*sizeof(uchar4);
	
	uchar4* d_sourceImg;
	uchar4* d_destImg;
	uchar4* d_finalImg;

	checkCudaErrors(cudaMalloc(&d_sourceImg, imageSize));
	checkCudaErrors(cudaMalloc(&d_destImg, 	 imageSize));
	checkCudaErrors(cudaMalloc(&d_finalImg,  imageSize));

  	checkCudaErrors(cudaMemcpy(d_sourceImg, h_sourceImg, imageSize, cudaMemcpyHostToDevice));
  	checkCudaErrors(cudaMemcpy(d_destImg, 	h_destImg, 	 imageSize, cudaMemcpyHostToDevice));

	// allocate predicate stuff
	size_t predicateSize = numRowsSource*numColsSource*sizeof(int);
	int* d_borderPredicate;
	int* d_interiorPredicate;

	checkCudaErrors(cudaMalloc(&d_borderPredicate, 	 predicateSize));
	checkCudaErrors(cudaMalloc(&d_interiorPredicate, predicateSize));

	// make reusable dims
	const dim3 blockSize(32, 32);
    const dim3 gridSize(numColsSource/blockSize.x + 1, numRowsSource/blockSize.y + 1);


	/* *****************************************************************
     1) Compute a mask of the pixels from the source image to be copied
        The pixels that shouldn't be copied are completely white, they
        have R=255, G=255, B=255.  Any other pixels SHOULD be copied.
        * *************************************************************/
	// first create mask
	unsigned char* d_mask; 

	checkCudaErrors(
		cudaMalloc( (void **) &d_mask, srcSize*sizeof(unsigned char)));
	checkCudaErrors(
		cudaMemset( d_mask, 0, srcSize*sizeof(unsigned char)) );

	create_mask<<<gridsize_mask, M_mask>>>( d_sourceImg, d_mask, srcSize);
	


	/**
     2) Compute the interior and border regions of the mask.  An interior
        pixel has all 4 neighbors also inside the mask.  A border pixel is
        in the mask itself, but has at least one neighbor that isn't.
	**/

	// generate the predicates
	maskPredicateKernel<<<gridSize, blockSize>>>(
		d_sourceImg,
		d_borderPredicate,
		d_interiorPredicate,
		numRowsSource,
		numColsSource
	);

 	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

 	/**
     3) Separate out the incoming image into three separate channels
	**/
	size_t floatSize = numRowsSource*numColsSource*sizeof(float);
	float *d_sourceImgR, *d_sourceImgG, *d_sourceImgB; 
	float *d_destImgR,   *d_destImgG, 	*d_destImgB;


	// split the source and destination images into their respective 
	//channels
	
	unsigned char* d_red_src ; 
	unsigned char* d_blue_src ; 
	unsigned char* d_green_src ; 

	checkCudaErrors(
		cudaMalloc( (void **) &d_red_src, srcSize*sizeof(unsigned char)));
	checkCudaErrors(
		cudaMalloc( (void **) &d_blue_src, srcSize*sizeof(unsigned char)));
	checkCudaErrors(
		cudaMalloc( (void **) &d_green_src, srcSize*sizeof(unsigned char)));
	checkCudaErrors(
		cudaMemset( d_red_src, 0, srcSize*sizeof(unsigned char)) );
	checkCudaErrors(
		cudaMemset( d_blue_src, 0, srcSize*sizeof(unsigned char)) );
	checkCudaErrors(
		cudaMemset( d_green_src, 0, srcSize*sizeof(unsigned char)) );

	separateChannels<<<gridsize_separate,M_separate>>>(d_sourceImg, 
										numColsSource, numRowsSource, 
										d_red_src,
										d_blue_src,
										d_green_src);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	
	

	checkCudaErrors(cudaMalloc(&d_sourceImgR, floatSize));
	checkCudaErrors(cudaMalloc(&d_sourceImgG, floatSize));
	checkCudaErrors(cudaMalloc(&d_sourceImgB, floatSize));
	
	checkCudaErrors(cudaMalloc(&d_destImgR, floatSize));
	checkCudaErrors(cudaMalloc(&d_destImgG, floatSize));
	checkCudaErrors(cudaMalloc(&d_destImgB, floatSize));
	
	separateChannelsKernel<<<gridSize, blockSize>>>(
		d_sourceImg,
		d_sourceImgR,
		d_sourceImgG,
		d_sourceImgB,
		numRowsSource,
		numColsSource);

 	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	separateChannelsKernel<<<gridSize, blockSize>>>(
		d_destImg,
		d_destImgR,
		d_destImgG,
		d_destImgB,
		numRowsSource,
		numColsSource);

 	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	/** 
     4) Create two float(!) buffers for each color channel that will
        act as our guesses.  Initialize them to the respective color
        channel of the source image since that will act as our intial guess.
	**/
	
	// allocate floats
	float *d_r0, *d_r1, *d_g0, *d_g1, *d_b0, *d_b1; 
	checkCudaErrors(cudaMalloc(&d_r0, floatSize));
	checkCudaErrors(cudaMalloc(&d_r1, floatSize));
	checkCudaErrors(cudaMalloc(&d_b0, floatSize));
	checkCudaErrors(cudaMalloc(&d_b1, floatSize));
	checkCudaErrors(cudaMalloc(&d_g0, floatSize));
	checkCudaErrors(cudaMalloc(&d_g1, floatSize));


  	checkCudaErrors(cudaMemcpy(d_r0, d_sourceImgR, floatSize, cudaMemcpyDeviceToDevice));
  	checkCudaErrors(cudaMemcpy(d_g0, d_sourceImgG, floatSize, cudaMemcpyDeviceToDevice));
  	checkCudaErrors(cudaMemcpy(d_b0, d_sourceImgB, floatSize, cudaMemcpyDeviceToDevice));

 	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	/**
     5) For each color channel perform the Jacobi iteration described 
        above 800 times.
	**/
	for(int i = 0; i < 800; i++) {
		jacobiKernel<<<gridSize, blockSize>>>(
			d_r0, 
			d_r1,
			d_borderPredicate,
			d_interiorPredicate,
			d_sourceImgR,
			d_destImgR,
			numRowsSource,
			numColsSource
		);
		std::swap(d_r0, d_r1);

		jacobiKernel<<<gridSize, blockSize>>>(
			d_g0, 
			d_g1,
			d_borderPredicate,
			d_interiorPredicate,
			d_sourceImgG,
			d_destImgG,
			numRowsSource,
			numColsSource
		);
		std::swap(d_g0, d_g1);

		jacobiKernel<<<gridSize, blockSize>>>(
			d_b0, 
			d_b1,
			d_borderPredicate,
			d_interiorPredicate,
			d_sourceImgB,
			d_destImgB,
			numRowsSource,
			numColsSource
		);
		std::swap(d_b0, d_b1);
	}

	/**
     6) Create the output image by replacing all the interior pixels
        in the destination image with the result of the Jacobi iterations.
        Just cast the floating point values to unsigned chars since we have
        already made sure to clamp them to the correct range.
	**/

	// lets assume that d_r0, d_g0, d_b0 are the final pass
/*	recombineChannelsKernel<<<gridSize, blockSize>>>(
		d_finalImg,
		d_r0,
		d_g0,
		d_b0,
		numRowsSource,
		numColsSource);
*/

	finalcopyChannels<<<gridsize_separate, M_separate>>>( d_finalImg, 
			numColsSource, numRowsSource, d_r0, d_g0, d_b0);

 	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	// copy device final image to host
  	checkCudaErrors(cudaMemcpy(h_blendedImg, d_finalImg, imageSize, cudaMemcpyDeviceToHost));

	// wow, we allocated a lot of (device GPU) memory!  cleanup, free device GPU memory
  	checkCudaErrors(cudaFree(d_sourceImg));
  	checkCudaErrors(cudaFree(d_destImg));
	checkCudaErrors(cudaFree(d_finalImg));

	checkCudaErrors(
		cudaFree( d_mask)) ;

	checkCudaErrors( cudaFree( d_red_src )); 
	checkCudaErrors( cudaFree( d_blue_src )); 
	checkCudaErrors( cudaFree( d_green_src )); 

	checkCudaErrors(cudaFree(d_sourceImgR));
	checkCudaErrors(cudaFree(d_sourceImgG));
	checkCudaErrors(cudaFree(d_sourceImgB));

	checkCudaErrors(cudaFree(d_destImgR));
	checkCudaErrors(cudaFree(d_destImgG));
	checkCudaErrors(cudaFree(d_destImgB));

	checkCudaErrors(cudaFree(d_r0));
	checkCudaErrors(cudaFree(d_r1));
	checkCudaErrors(cudaFree(d_g0));
	checkCudaErrors(cudaFree(d_g1));
	checkCudaErrors(cudaFree(d_b0));
	checkCudaErrors(cudaFree(d_b1));
}
