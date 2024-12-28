/*
 * SSR 0.1 -- Applying Simple Sampling Reduction on CPU and Nvidia CUDA enabled devices --
 * Copyright (C) 2011 Michael Schneider, Norman Lahr

 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.

 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.

 * Contact: norman@lahr.email, mischnei@cdc.informatik.tu-darmstadt.de

 * Refer to the file LICENSE for the details of the GPL.
 *
 */

#ifndef _SAMPLINGCUDA_KERNEL_H_
#define _SAMPLINGCUDA_KERNEL_H_

#include <stdio.h>

#include <shrUtils.h>
#include "cutil_inline.h"
#include "cuda_runtime.h"

#include <limits.h>
#include <common_functions.h>

/* Include common constants */
#include "include/common/constants.h"
/* Include a pseudo-random number generator */
#include "cuda/PRNG.cu"
/* Include the cuda print function */
#include "cuda/cuPrintf.cu"

#if __CUDA_ARCH__ < 200 	//Compute capability 1.x architectures
#define CUPRINTF cuPrintf
#else						//Compute capability 2.x architectures
#define CUPRINTF(fmt, ...) printf("[%d, %d]:\t" fmt, \
								blockIdx.y*gridDim.x+blockIdx.x,\
								threadIdx.z*blockDim.x*blockDim.y+threadIdx.y*blockDim.x+threadIdx.x,\
								__VA_ARGS__)
#endif

/* Include texture definitions */
#include "include/cuda/SamplingCUDA_Textures.h"

#define __XY(x,y) ((x)*const_Dim+(y))

extern __shared__ int sharedArray[];

/**
 * Kernel for calculating a number of shorter vecotr v's with the sample algorithm.
 */
__global__ void sampleKernel(typeV *resV, unsigned long *resNormV, unsigned int* counter){

/* Initialize block's norm */
resNormV[blockIdx.x] = 0L;

/* If the counter is set to zero by the atomicInc() operation,
 * then a shorter vector was found. */
if((((*counter) <= const_numAddMax) && const_firstTime) || (((*counter) <= 1) && !const_firstTime)){
	/* Mapping of dynamically allocated shared memory */
	char*	valid = (char*) sharedArray;
	int* 	y = (int*) &valid[const_SamplesPerBlock];
	typeV*	v = (typeV*) &y[const_SamplesPerBlock];
	float*	mu = (float*) &v[const_SamplesPerBlock*const_Dim];

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int i = 0,
			j = 0,
			k = 0,
			kTemp = 0;

	long temp = 0L;

	/* If dim is greater than the available threads per block, this is the
	 * number of parts in which a parallel operation is split. */
	const int kMax = (int) ceilf((float)const_Dim/(float)blockDim.x);

	/* Set identity of one Sample and initialize the valid byte*/
	unsigned long x = 0L;
	if(tid < const_SamplesPerBlock){
		x =  const_xOffset + bid * const_SamplesPerBlock + tid;
		valid[tid] = 1; // all valid
	}


	/* Copy last vector of B and R to v and mu */
	for(i = 0; i < const_SamplesPerBlock; i++){
		__syncthreads();
		for(k = 0; k < kMax; k++){
			__syncthreads();
			kTemp = k * blockDim.x + tid;
			if(kTemp < const_Dim){
				v[__XY(i,kTemp)] = tex1Dfetch(texRefB, __XY((const_Dim-1),kTemp));
				mu[__XY(i,kTemp)]= tex1Dfetch(texRefR, __XY((const_Dim-1),kTemp));
			}
		}
	}

	for(i = const_Dim-2; i >= 0; i--){
		__syncthreads();

		/* determine y's */
		if(tid < const_SamplesPerBlock){
			y[tid] = (int) ceilf(mu[__XY(tid,i)] - 0.5);

//			y[tid] = y[tid] + ((((mu[__XY(tid,i)] - (float)y[tid]) > 0.0)-1) | 1) * (x & 1L);
			if( x % 2L == 1L){
				if(mu[__XY(tid,i)] - (float)y[tid] <= 0.0){
					y[tid] -= 1;
				}
				else{
					y[tid] += 1;
				}

			}

			x = x >> 1; // x/2
		}

		/* Determining a new vector v */
		for(j = 0; j < const_SamplesPerBlock; j++){
			__syncthreads();
			if(y[j] != 0 && valid[j] != 0)
				for(k = 0; k < kMax; k++){
					__syncthreads();
					kTemp = k * blockDim.x + tid;
					if(kTemp < const_Dim){
						if(y[j] == 1){
							temp = v[__XY(j, kTemp)] - tex1Dfetch(texRefB, __XY(i, kTemp));
							/* Test for overflow*/
							if(temp >= INT_MIN){
								v[__XY(j, kTemp)] = temp;
								mu[__XY(j, kTemp)] = mu[__XY(j, kTemp)] - (float)tex1Dfetch(texRefR, __XY(i, kTemp));
							}
							else{
								/* Disable the sample, because of an overflow */
								valid[j] = 0;
							}
						}
						else if(y[j] == -1){
							temp = v[__XY(j, kTemp)] + tex1Dfetch(texRefB, __XY(i, kTemp));
							/* Test for overflow*/
							if(temp <= INT_MAX){
								v[__XY(j, kTemp)] = temp;
								mu[__XY(j, kTemp)] = mu[__XY(j, kTemp)] + (float)tex1Dfetch(texRefR, __XY(i, kTemp));
							}
							else{
								/* Disable the sample, because of an overflow */
								valid[j] = 0;
							}
						}
						else{
							temp = v[__XY(j, kTemp)] - y[j] * tex1Dfetch(texRefB, __XY(i, kTemp));

							/* Test for overflow*/
							if(temp >= INT_MIN && temp <= INT_MAX){
								v[__XY(j, kTemp)] = temp;
								mu[__XY(j, kTemp)] = mu[__XY(j, kTemp)] - (float)y[j] * (float)tex1Dfetch(texRefR, __XY(i, kTemp));
							}
							else{
								/* Disable the sample, because of an overflow */
								valid[j] = 0;
							}
						}
					}
				}
		}
		__syncthreads();
	}

	__syncthreads();
	/* calc the square norms in parallel */
	/* Use space in shared memory from mu, because it is not longer needed.
	 * mu used samplesPerBlock * dim * 4Byte, norms uses dim * 8 Bytes.
	 * If there are more than 2 samples per block, norms fits in the space of mu.*/
	unsigned long* norms  = (unsigned long*) &v[const_SamplesPerBlock*const_Dim];
	int dimPow2,	/* Next 2^x */
		offStart,	/* Offset of the actual interval */
		num;		/* Number of the elements left */

	unsigned long minNorm = ULONG_MAX,	/* Minimum of a block */
			minIdx = 0;					/* Index of the minimal sample */

	unsigned int stride;

	/* Calc norm for every sample in a block */
	for(i = 0; i < const_SamplesPerBlock; i++){
		__syncthreads();
		/* Check if the actual sample is valid, if not go to next sample and gain performance */
		if(valid[i] != 0){
			/* Number of elements for norm calculation is the size of the dimension */
			num = const_Dim;

			offStart = 0;

			/* Square all entries in parallel */
			for(k = 0; k < kMax; k++){
				__syncthreads();
				kTemp = k * blockDim.x + tid;
				if(kTemp < const_Dim){
					norms[kTemp] = (unsigned long)v[__XY(i, kTemp)] * (unsigned long)v[__XY(i, kTemp)];
				}
			}

#ifdef DEBUG
			/* Control norm */
			__syncthreads();
			unsigned long serialNorm = 0;
			for(j=0; j<const_Dim; j++){
				serialNorm += norms[j];
			}
#endif

			/* Add dimPow2 elements in each step */
			for(dimPow2 = __powf(2, (int)__log2f(num)); num > 0; dimPow2 = __powf(2, (int)__log2f(num))){
				__syncthreads();

				num -= dimPow2;
				/* Add dimPow2 elements in log(dimPow2) steps */
				for (stride = dimPow2 ; stride > 1; ) {
					__syncthreads();
					stride = stride >> 1;
					for(k = 0; k < kMax; k++){
						__syncthreads();
						kTemp = k * blockDim.x + tid;
						if(kTemp < const_Dim){
							if (kTemp < stride)
								norms[kTemp + offStart] += norms[kTemp + offStart + stride] ;
						}
					}
				}
				/* Add the partial sum in the first element to the first element of the following block */
				__syncthreads();
				if(tid == 0)
					norms[offStart + dimPow2] += norms[offStart];

				__syncthreads();
				if(num > 0)
					offStart += dimPow2;
			}
			__syncthreads();


			if(norms[offStart] < minNorm){
				minNorm = (unsigned long)norms[offStart];
				minIdx = i;
			}
#ifdef DEBUG
			if(serialNorm != norms[offStart])
				CUPRINTF("Check Norm calculation...\t\tFAILED!\tSample %i\tBid: %i\tSerialNorm: %u\tParallelNorm: %u\n",i,bid,serialNorm, norms[offStart]);
#endif
		}
	}
	__syncthreads();
	/* Find minimum of all square norms and copy the smallest to host */
	if(tid == 0){
		if(minNorm != ULONG_MAX && minNorm < 0.99 * const_normB1){
			resNormV[bid] = (unsigned long)minNorm;
			/* Increase the abortion counter, if a shorter vector was found */
			atomicAdd(counter, 1);
		}
		else{
			resNormV[bid] = 0L;
		}

	}

	for(k = 0; k < (int) ceilf((float)const_Dim/(float)blockDim.x); k++){
		__syncthreads();
		kTemp = k * blockDim.x + tid;
		if(kTemp < const_Dim){
			resV[__XY(bid,kTemp)] = v[__XY(minIdx, kTemp)];
		}
	}

	__syncthreads();
}
}

#define RAND_NUM_DISCRETE_VALS 1000
/**
 * Kernel for calculating a number of shorter vecotr v's with the sampling algorithm.
 */
__global__ void samplingKernel(typeV *resV, unsigned long *resNormV){

	/* Mapping of dynamically allocated shared memory */
	char*	valid = (char*) sharedArray;
	int* 	y = (int*) &valid[const_SamplesPerBlock];
	typeV*	v = (typeV*) &y[const_SamplesPerBlock];
	float*	mu = (float*) &v[const_SamplesPerBlock*const_Dim];

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int i = 0,
			j = 0,
			k = 0,
			kTemp = 0;

	long temp = 0L;

	float decPlace = 0.0, tempFloat = 0.0;
	int tempInt = 0;

	int idums;

	/* If dim is greater than the available threads per block, this is the
	 * number of parts in which a parallel operation is split. */
	const int kMax = (int) ceilf((float)const_Dim/(float)blockDim.x);

	/* Set identity of one Sample and initialize the valid byte*/
	long x = 0L;
	if(tid < const_SamplesPerBlock){
		x = (long) const_xOffset + bid * const_SamplesPerBlock + tid;
		valid[tid] = 1; // all valid
	}


	/* Copy last vector of B and R to v and mu */
	for(i = 0; i < const_SamplesPerBlock; i++){
		__syncthreads();
		for(k = 0; k < kMax; k++){
			__syncthreads();
			kTemp = k * blockDim.x + tid;
			if(kTemp < const_Dim){
				v[__XY(i,kTemp)] = tex1Dfetch(texRefB, __XY((const_Dim-1),kTemp));
				mu[__XY(i,kTemp)]= tex1Dfetch(texRefR, __XY((const_Dim-1),kTemp));
			}
		}
	}

	for(i = const_Dim-2; i >= 0; i--){
		__syncthreads();

		/* determine y's */
		if(tid < const_SamplesPerBlock){

			/* filter decimal place */
			tempFloat = mu[__XY(tid,i)];
			tempInt = (int)mu[__XY(tid,i)];
			decPlace = fabsf(tempFloat - tempInt);

			if((long)i < (long)const_Dim - (x%const_Dim)){
				/* search for a y that fits into |mu-y| <= 0.5 */

				/*
				 * There is no need for a random y in general,
				 * because there is only one y that fits.
				 * There is only the case, if the decimal place of
				 * Mu is equal 0.5. In this case y can be two values,
				 * which will estimated randomly.
				 */

				/* estimate y */
				if(decPlace < 0.5){
					y[tid] = tempInt;
				}
				else if(decPlace > 0.5){
					if(tempFloat < 0.0){
						y[tid] = tempInt - 1;
					}
					else{
						y[tid] = tempInt + 1;
					}
				}
				else if(decPlace == 0.5){
					if(rand_Discrete((int*)&x, &idums, RAND_NUM_DISCRETE_VALS) % 2 == 0){
						y[tid] = tempInt;
					}
					else{
						if(tempFloat < 0.0){
							y[tid] = tempInt - 1;
						}
						else{
							y[tid] = tempInt + 1;
						}
					}
				}
			}
			else{

				/* search for a y that fits into |mu-y| <= 1 */

				/* estimate y */
				if(decPlace != 0.0){
					if(rand_Discrete((int*)&x, &idums, RAND_NUM_DISCRETE_VALS) % 2 == 0){
						y[tid] = tempInt;
					}
					else{
						if(tempFloat < 0.0){
							y[tid] = tempInt - 1;
						}
						else{
							y[tid] = tempInt + 1;
						}
					}
				}
				else{
					switch (rand_Discrete((int*)&x, &idums, RAND_NUM_DISCRETE_VALS) % 3) {
						case 0:
							y[tid] = tempInt - 1;
							break;
						case 1:
							y[tid] = tempInt;
							break;
						case 2:
							y[tid] = tempInt + 1;
							break;
						default:
							break;
					}
				}
			}
		}

		/* Determining a new vector v */
		for(j = 0; j < const_SamplesPerBlock; j++){
			__syncthreads();
			if(y[j] != 0 && valid[j] != 0)
				for(k = 0; k < kMax; k++){
					__syncthreads();
					kTemp = k * blockDim.x + tid;
					if(kTemp < const_Dim){
						if(y[j] == 1){
							temp = v[__XY(j, kTemp)] - tex1Dfetch(texRefB, __XY(i, kTemp));
							/* Test for overflow*/
							if(temp >= INT_MIN){
								v[__XY(j, kTemp)] = temp;
								mu[__XY(j, kTemp)] = mu[__XY(j, kTemp)] - (float)tex1Dfetch(texRefR, __XY(i, kTemp));
							}
							else{
								/* Disable the sample, because of an overflow */
								valid[j] = 0;
							}
						}
						else if(y[j] == -1){
							temp = v[__XY(j, kTemp)] + tex1Dfetch(texRefB, __XY(i, kTemp));
							/* Test for overflow*/
							if(temp <= INT_MAX){
								v[__XY(j, kTemp)] = temp;
								mu[__XY(j, kTemp)] = mu[__XY(j, kTemp)] + (float)tex1Dfetch(texRefR, __XY(i, kTemp));
							}
							else{
								/* Disable the sample, because of an overflow */
								valid[j] = 0;
							}
						}
						else{
							temp = v[__XY(j, kTemp)] - y[j] * tex1Dfetch(texRefB, __XY(i, kTemp));

							/* Test for overflow*/
							if(temp >= INT_MIN && temp <= INT_MAX){
								v[__XY(j, kTemp)] = temp;
								mu[__XY(j, kTemp)] = mu[__XY(j, kTemp)] - (float)y[j] * (float)tex1Dfetch(texRefR, __XY(i, kTemp));
							}
							else{
								/* Disable the sample, because of an overflow */
								valid[j] = 0;
							}
						}
					}
				}
		}
		__syncthreads();
	}

	__syncthreads();
	/* calc the square norms in parallel */
	/* Use space in shared memory from mu, because it is not longer needed.
	 * mu used samplesPerBlock * dim * 4Byte, norms uses dim * 8 Bytes.
	 * If there are more than 2 samples per block, norms fits in the space of mu.*/
	unsigned long* norms  = (unsigned long*) &v[const_SamplesPerBlock*const_Dim];
	int dimPow2,	/* Next 2^x */
		offStart,	/* Offset of the actual interval */
		num;		/* Number of the elements left */

	unsigned long minNorm = ULONG_MAX,	/* Minimum of a block */
			minIdx = 0;					/* Index of the minimal sample */

	unsigned int stride;

	/* Calc norm for every sample in a block */
	for(i = 0; i < const_SamplesPerBlock; i++){
		__syncthreads();
		/* Check if the actual sample is valid, if not go to next sample and gain performance */
		if(valid[i] != 0){
			/* Number of elements for norm calculation is the size of the dimension */
			num = const_Dim;

			offStart = 0;

			/* Square all entries in parallel */
			for(k = 0; k < kMax; k++){
				__syncthreads();
				kTemp = k * blockDim.x + tid;
				if(kTemp < const_Dim){
					norms[kTemp] = (unsigned long)v[__XY(i, kTemp)] * (unsigned long)v[__XY(i, kTemp)];
				}
			}

#ifdef DEBUG
			/* Control norm */
			__syncthreads();
			unsigned long serialNorm = 0;
			for(j=0; j<const_Dim; j++){
				serialNorm += norms[j];
			}
#endif

			/* Add dimPow2 elements in each step */
			for(dimPow2 = __powf(2, (int)__log2f(num)); num > 0; dimPow2 = __powf(2, (int)__log2f(num))){
				__syncthreads();

				num -= dimPow2;
				/* Add dimPow2 elements in log(dimPow2) steps */
				for (stride = dimPow2 ; stride > 1; ) {
					__syncthreads();
					stride = stride >> 1;
					for(k = 0; k < kMax; k++){
						__syncthreads();
						kTemp = k * blockDim.x + tid;
						if(kTemp < const_Dim){
							if (kTemp < stride)
								norms[kTemp + offStart] += norms[kTemp + offStart + stride] ;
						}
					}
				}
				/* Add the partial sum in the first element to the first element of the following block */
				__syncthreads();
				if(tid == 0)
					norms[offStart + dimPow2] += norms[offStart];

				__syncthreads();
				if(num > 0)
					offStart += dimPow2;
			}
			__syncthreads();


			if(norms[offStart] < minNorm){
				minNorm = (unsigned long)norms[offStart];
				minIdx = i;
			}
#ifdef DEBUG
			if(serialNorm != norms[offStart])
				CUPRINTF("Check Norm calculation...\t\tFAILED!\tSample %i\tBid: %i\tSerialNorm: %u\tParallelNorm: %u\n",i,bid,serialNorm, norms[offStart]);
#endif
		}
	}
	__syncthreads();
	/* Find minimum of all square norms and copy the smallest to host */
	if(tid == 0){
		if(minNorm != ULONG_MAX && minNorm < 0.99 * const_normB1){
			resNormV[bid] = (unsigned long)minNorm;
		}
		else{
			resNormV[bid] = 0L;
		}

	}

	for(k = 0; k < (int) ceilf((float)const_Dim/(float)blockDim.x); k++){
		__syncthreads();
		kTemp = k * blockDim.x + tid;
		if(kTemp < const_Dim){
			resV[__XY(bid,kTemp)] = v[__XY(minIdx, kTemp)];
		}
	}

	__syncthreads();
}


#endif
