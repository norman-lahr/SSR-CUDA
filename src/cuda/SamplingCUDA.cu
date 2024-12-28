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

/**
 * Host part of the CUDA interface.
 * It will be compiled with CUDA compiler.
 */

/* Include system libs*/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>

/* Loggin feature */
#include "include/common/Log.h"

/* CUDA Utilities */
#include <cutil_inline.h>

/* Include defined constants */
#include "include/common/constants.h"

/* Include texture definitions */
#include "include/cuda/SamplingCUDA_Textures.h"

/* Include Kernels */
#include "cuda/SamplingCUDA_Kernel.cu"


/**
 * Interface function, which is compiled with CUDA compiler and
 * called by the host code which is compiled by GCC.
 */
extern "C"
void sample(typeV* shortV, short* shortB, float* floatR, int dim,
		unsigned long normB1, unsigned long *normV,
		unsigned long samplesPerRound,
		int deviceID, int numBlocks, unsigned long xOffset, int samplingAlgorithm, int numAddMax){

	static bool firstTime = true;

	/* Choose the device by deviceID */
	cutilSafeCall(cudaSetDevice(deviceID));
	/* Get device properties */
	cudaDeviceProp devProps;
	cudaGetDeviceProperties(&devProps, deviceID);

   /* Calculate maximal number of samples per block,
    * depending on available shared memory.
    * Vectors v and mu resides in this kind of memory.*/
	/* size of v, mu, y and valid */
	unsigned int usedSharedMemPerSample = ((sizeof(typeV) + sizeof(float)) * dim) + sizeof(int) + sizeof(char);
	unsigned int samplesPerBlock = (AVAILABLE_SHARED_MEM_SAMPLEKERNEL / usedSharedMemPerSample);

	/* Determine number of threads per Block */
	int threadsPerBlock;
	if (dim < devProps.maxThreadsPerBlock)
	   threadsPerBlock = dim;
	else
	   threadsPerBlock = devProps.maxThreadsPerBlock;

	/* Memory initializations.*/

	short *devB;
	float *devR;
	typeV *devV;

	unsigned long *devNorm;

	unsigned int *devCounter;
	unsigned int hostCounter = 0;

	unsigned int sizeB = dim * dim * sizeof(short);
	unsigned int sizeR = dim * dim * sizeof(float);
	unsigned int sizeV = numBlocks * dim * sizeof(typeV);
	unsigned int d_sizeNormV = numBlocks * sizeof(unsigned long);

	unsigned int usedSharedMem = 0;

	cutilSafeCall(cudaMalloc((void**)&devB, sizeB));
	cutilSafeCall(cudaMalloc((void**)&devR, sizeR));
	cutilSafeCall(cudaMalloc((void**)&devV, sizeV));
	cutilSafeCall(cudaMalloc((void**)&devNorm, d_sizeNormV));
	cutilSafeCall(cudaMalloc((void**)&devCounter, sizeof(unsigned int)));

	cutilSafeCall(cudaMemcpy(devB, shortB, sizeB, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(devR, floatR, sizeR, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(devCounter, &hostCounter, sizeof(unsigned int), cudaMemcpyHostToDevice));

	/* Bind both matrices to textures */
	cudaBindTexture(0, texRefB, devB, sizeB);
	cudaBindTexture(0, texRefR, devR, sizeR);

	cudaMemcpyToSymbol(const_Dim, &dim, sizeof(dim));
	cudaMemcpyToSymbol(const_SamplesPerBlock, &samplesPerBlock, sizeof(samplesPerBlock));
	cudaMemcpyToSymbol(const_xOffset, &xOffset, sizeof(xOffset));
	cudaMemcpyToSymbol(const_normB1, &normB1, sizeof(normB1));
	cudaMemcpyToSymbol(const_firstTime, &firstTime, sizeof(firstTime));
	cudaMemcpyToSymbol(const_numAddMax, &numAddMax, sizeof(numAddMax));

#ifdef DEBUG
	/* Initialize cudaPrint */
	cudaPrintfInit();
#endif

	/* setup execution parameters for sampling kernel*/
	dim3 dimBlock1(threadsPerBlock,1,1);
	dim3 dimGrid1(numBlocks,1,1);

	/* execute kernel for sampling, align AVAILABLE_SHARED_MEM_SAMPLEKERNEL in constants.h!
	 * samplesPerBlock > 2 !
	 */
	usedSharedMem =  usedSharedMemPerSample * samplesPerBlock;

	switch (samplingAlgorithm) {
		case SAMPLE_ALGORITHM:
				sampleKernel<<< dimGrid1, dimBlock1 , usedSharedMem >>>((typeV*)devV, (unsigned long*)devNorm, devCounter);
			break;
		case SAMPLING_ALGORITHM:
				samplingKernel<<< dimGrid1, dimBlock1 , usedSharedMem >>>((typeV*)devV, (unsigned long*)devNorm);
			break;
		default:
			break;
	}

	cudaThreadSynchronize();

#ifdef DEBUG
	/* Write out messages from device */
	cudaPrintfDisplay(stdout, false);
	cudaPrintfEnd();
#endif

   /* check if kernel execution generated and error */
   cutilCheckMsg("Kernel execution failed");

   /* Get results from the device */
   cutilSafeCall(cudaMemcpy(normV, (unsigned long*)devNorm, d_sizeNormV, cudaMemcpyDeviceToHost));
   cutilSafeCall(cudaMemcpy(shortV, (typeV*)devV, sizeV, cudaMemcpyDeviceToHost));

   /* Free device memory */
   cudaFree(devB);
   cudaFree(devR);
   cudaFree(devV);
   cudaFree(devNorm);

   cudaThreadExit();

   firstTime = false;
}
