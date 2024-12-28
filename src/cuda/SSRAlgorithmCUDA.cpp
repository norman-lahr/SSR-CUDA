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

#include "include/cuda/SSRAlgorithmCUDA.h"

/* Forwarding to CUDA compiler */
extern "C"
void sample(typeV* shortV, short* shortB, float* floatR, int dim,
		unsigned long normB1, unsigned long *normV,
		unsigned long samplesPerRound,
		int deviceID, int numBlocks, unsigned long xOffset, int samplingAlgorithm, int numAddMax);

SSRAlgorithmCUDA::SSRAlgorithmCUDA(mat_ZZ B, int reducingAlg, int initBlocksize, int inLoopBlocksize, float delta, int initPrecision, int inLoopPrecision, float goalNorm, string filename){

	this->B = B;
	this->G.SetDims(B.NumRows(),B.NumCols());

	this->reducingAlg = reducingAlg;
	this->initBlocksize = initBlocksize;
	this->inLoopBlocksize = inLoopBlocksize;
	this->delta = delta;
	this->initPrecision = initPrecision;
	this->inLoopPrecision = inLoopPrecision;
	this->goalNorm =(unsigned long) (goalNorm * goalNorm);

	this->filename = filename;
}

SSRAlgorithmCUDA::~SSRAlgorithmCUDA(){

}

/**
 *
 */
mat_ZZ SSRAlgorithmCUDA::run(int SampleAlgorithm, int startIter, unsigned long stopIter, int deviceID, int numBlocks, int numAddVMax){

	Log::info("SSR algorithm on GPU will produce " + Log::toStr(stopIter) + " Samples ...");

	/* Offset of actual remaining x */
	unsigned long xOffset = 0;

	/* Remain number of samples */
	long remainSamples = 0L;

	/* Samples per CUDA call */
	unsigned long samplesPerRound = 0;

	/* Gram-Schmidt coefficient matrix */
	mat_RR R;

	/* Lengths of orthogonal vectors */
	vec_RR b;

	/* Shorter vector*/
	vec_ZZ v;
	v.SetLength(this->B.NumRows());

	/* Short variant of lattice Basis B for CUDA */
	short *shortB;

	/* Short variant of Gram-Schmidt coefficient matrix for CUDA */
	float *floatR;

	/* Short variant of resulting vector v for CUDA */
	typeV *shortV;

	/* Euclidean Norm of first basis vector */
	unsigned long normB1 = 0UL;

	/* Euclidean Norm of vector v */
	unsigned long *normV = (unsigned long*)malloc(numBlocks * this->B.NumCols() * sizeof(unsigned long));

	/* Smallest v and its Norm */
	vec_ZZ minV;
	unsigned long minNormV = ULONG_MAX;

	/* Number of v's, which are added to G */
	int numAddV = 0;

	/* found a smaller vector v */
	bool foundSmaller = true;

	/* Times for time measurement */
	struct timeval start, end;
	long mtime = 1L, seconds = 1L, useconds = 1L;

	int j = 0;
	long numSamples = 0L;

	/* File output stream */
	ofstream outputFile;

	/* Apply Reducing-Algorithm */
	Log::info("Apply first LLL ...");
	NTL::LLL_RR(this->B, delta);
	if(this->reducingAlg != __LLL){
		Log::info("Apply first BKZ ...");
		switch (this->initPrecision) {
			case 1:
				NTL::BKZ_FP(this->B, delta, this->initBlocksize);
				break;
			case 2:
				NTL::BKZ_QP(this->B, delta, this->initBlocksize);
				break;
			case 3:
				NTL::BKZ_XD(this->B, delta, this->initBlocksize);
				break;
			case 4:
				NTL::BKZ_RR(this->B, delta, this->initBlocksize);
				break;
			default:
				break;
		}
	}
	/* Compute Gram-Schmidt decomposition */
	NTL::ComputeGS(this->B, R, b);

	/* Initialize matrices and the vector */
	shortB = (short*) malloc(this->B.NumCols() * this->B.NumRows() * sizeof(short));
	floatR = (float*) malloc(R.NumCols() * R.NumRows() * sizeof(float));
	shortV = (typeV*) malloc(numBlocks * this->B.NumCols() * sizeof(typeV));

	/* Read out device properties */
	cudaDeviceProp devProps;
	cudaGetDeviceProperties(&devProps, deviceID);
	Log::info("Using CUDA device [" + Log::toStr(deviceID) + "]: " + string(devProps.name));

	/* Calc the maximum of concurrent sample
	 * temp: sharedMemorySize / 6 * (Dim of B), v and mu are stored in shared memory
	* max samples per block: sharedMemorySize / (6 * (Dim of B) + temp * 4), y is also stored in shared memory
	* max samples per round: (max samples per block) * (max number of block)
	*/
	/* size of v, mu, y and valid */
	unsigned int usedSharedMemPerSample = ((sizeof(typeV) + sizeof(float)) * this->B.NumCols()) + sizeof(int) + sizeof(char);
	unsigned int samplesPerBlock = (AVAILABLE_SHARED_MEM_SAMPLEKERNEL / usedSharedMemPerSample);
	long maxSamplesPerRound = samplesPerBlock * numBlocks;

	/* Parallel Norm Calculation need a minimum of 2 mu's in one block. */
	if(samplesPerBlock < 2){
		Log::error("Dimension is too high, not enough Shared Memory space!");
		exit(-1);
	}

#ifdef DEBUG
	Log::debug("Shared Mem per Block: " + Log::toStr(devProps.sharedMemPerBlock)+
			"\nAvailable Shared Mem for SampleKernel: " + Log::toStr(AVAILABLE_SHARED_MEM_SAMPLEKERNEL)+
			"\nSizeOf Short: " + Log::toStr(sizeof(short)) + ", SizeOf Float: " + Log::toStr(sizeof(float)) + ", SizeOf Int: " + Log::toStr(sizeof(int)) + ", Sizeof Long: " + Log::toStr(sizeof(long))+
			"\nDim of B: " + Log::toStr(this->B.NumCols())+
			"\nMaxThreadPerBlock: " + Log::toStr(devProps.maxThreadsDim[0])+
			"\nMaxBlocks: " + Log::toStr(numBlocks)+
			"\nusedSharedMemPerSample: " + Log::toStr(usedSharedMemPerSample)+
			"\nsamplesPerBlock: " + Log::toStr(samplesPerBlock)+
			"\nmaxSamplesPerRound: " + Log::toStr(maxSamplesPerRound));
#endif

	/* Calculate the norm of the first vector in lattice B, before sampling starts */
	euclidNormSqr(&normB1, &this->B[0]);

	/* Search for a smaller vector */
	Log::info("Start searching for new vectors with SSR algorithm ...");
	while (foundSmaller && (this->goalNorm < normB1)){

		foundSmaller = false;

		/* Reset */
		xOffset = 0;

		Log::info("Norm(B[0]) = " + Log::toStr((long)sqrt(normB1)));

		//////////////////////////////////////////////////////
		// Run sampling with CUDA
		//////////////////////////////////////////////////////

		/* Transform from NTL format to usual format */
		toShortMat(shortB, &this->B);
		toFloatMat(floatR, &R);

		/* Run SSR algorithm */
		for(remainSamples = (stopIter - startIter); remainSamples > 0L && !foundSmaller; remainSamples -= maxSamplesPerRound){

			/* Determine the number of samples per round,
			 * depending on the remaining samples */
			if (remainSamples > maxSamplesPerRound)
				samplesPerRound = maxSamplesPerRound;
			else
				samplesPerRound = remainSamples;

			/* measure the starting time */
			gettimeofday(&start, NULL);

			/* Produce a shorter vector than normB1 or nothing */
			sample(shortV, shortB, floatR, this->B.NumCols(),
					normB1, normV,
					samplesPerRound,
					deviceID, numBlocks, xOffset, SampleAlgorithm, numAddVMax);

			/* Time measurement */
			gettimeofday(&end, NULL);
			seconds  = end.tv_sec  - start.tv_sec;
			useconds = end.tv_usec - start.tv_usec;
			mtime = seconds*1000 + useconds/1000;

			/* Increase the offset of variable x */
			xOffset += samplesPerRound;

			   /* Check if the calculated square norm is smaller
			    * than 99% of the delivered square norm of the
			    * first vector of B */
			foundSmaller = false;
			for(j = 0; j < numBlocks && !foundSmaller; j++){
				if(normV[j] != 0L)
					foundSmaller = true;
			}


			/* Output the actual progress and samplerate continuously */
			Log::info("Progress: " + Log::toStr((int)ceil((double)xOffset/(double)stopIter*100.0)) + "%" + " (" + Log::toStr(xOffset) + " Samples)" + "\t" +
					"Samplerate: " + Log::toStr((int)(((double)numSamples)/(double)((double)mtime/(double)1000))) + " Samples/sec", false, true, true);

			/* For calculating the current samplerate */
			numSamples = maxSamplesPerRound;

		}

		/////////////////////////////////////////////////////
		// End of "CUDA Domain"
		/////////////////////////////////////////////////////

		if(foundSmaller){
			/* If the for-loop breaks because of a smaller vector v */
			/* insert v into G */
			/* Set v as first columns vector in lattice G */

			AddMinVecsToG(shortV, normV, &numAddV, numAddVMax, &minNormV, &minV, numBlocks, &numSamples);

			Log::info("Found " + Log::toStr(numAddV) + " smaller vector(s)!" + "\n\t\t\t" +
					"Norm([Smallest vector]) = " + Log::toStr((long)sqrt(minNormV)) + " (" + Log::toStr((long)(((double)minNormV/(double)normB1)*100.0)-100L) + "%)" + "\n\t\t\t" +
					"[Smallest vector] = " + Log::toStr(minV) + "\n");

			/* Apply Reducing-Algorithm */
			if(this->reducingAlg != __LLL){
				Log::info("Apply BKZ ...");
				switch (this->inLoopPrecision) {
					case 1:
						NTL::BKZ_FP(this->G, delta, this->initBlocksize);
						break;
					case 2:
						NTL::BKZ_QP(this->G, delta, this->initBlocksize);
						break;
					case 3:
						NTL::BKZ_XD(this->G, delta, this->initBlocksize);
						break;
					case 4:
						NTL::BKZ_RR(this->G, delta, this->initBlocksize);
						break;
					default:
						break;
				}
			}
			else{
				Log::info("Apply LLL ...");
				switch (this->inLoopPrecision) {
					case 1:
						NTL::LLL_FP(this->G, delta);
						break;
					case 2:
						NTL::LLL_QP(this->G, delta);
						break;
					case 3:
						NTL::LLL_XD(this->G, delta);
						break;
					case 4:
						NTL::LLL_RR(this->G, delta);
						break;
					default:
						break;
				}
			}

			RemZeroVecs();

			/* Save results */
			outputFile.open(this->filename.c_str());

			if (!outputFile){
				Log::error("Output-File could not opened!");
			}
			else{
				outputFile << this->B;
				outputFile.close();
			}

			/* Compute Gram-Schmidt decomposition */
			NTL::ComputeGS(this->B, R, b);
		}

		/* Calculate Norm for abort criterion */
		euclidNormSqr(&normB1, &B[0]);

	}

	unsigned long int norm = 0;
	euclidNormSqr(&norm, &B[0]);
	Log::printNorm((long)sqrt(norm));

	free(shortB);
	free(floatR);
	free(shortV);

	shortB = NULL;
	floatR = NULL;
	shortV = NULL;

	return this->B;
}

void SSRAlgorithmCUDA::euclidNormSqr(unsigned long *res, vec_ZZ *vec){

	ZZ tempRes = to_ZZ(0);

	for(int i = 0; i < vec->length(); i++){
		tempRes += (*vec)[i] * (*vec)[i];
	}

	if(tempRes > to_ZZ(ULONG_MAX)){
		Log::error("euclidNormSqr: Norm of vector is too big for type unsigned long!\nSearching canceled.");
		exit(-1);
	}
	*res = (unsigned long)to_ulong(tempRes);
}

void SSRAlgorithmCUDA::toShortMat(short *resMat, mat_ZZ *mat){
	for(int i = 0; i < mat->NumRows(); i++){
		for(int j = 0; j < mat->NumCols(); j++){
			resMat[i * mat->NumCols() + j] = checkForShort(to_int((*mat)[i][j]));
		}
	}
}

void SSRAlgorithmCUDA::toFloatMat(float *resMat, mat_RR *mat){
	for(int i = 0; i < mat->NumRows(); i++){
		for(int j = 0; j < mat->NumCols(); j++){
			resMat[i * mat->NumCols() + j] = to_float((*mat)[i][j]);
		}
	}
}

void SSRAlgorithmCUDA::toShortVec(typeV *resVec, vec_ZZ *vec){
	for(int i = 0; i < vec->MaxLength(); i++){
		resVec[i] = checkForShort(to_int((*vec)[i]));
	}
}

void SSRAlgorithmCUDA::toZZMat(short *resMat, mat_ZZ *mat){
	for(int i = 0; i < mat->NumRows(); i++){
		for(int j = 0; j < mat->NumCols(); j++){
			(*mat)[i][j] = to_ZZ(resMat[i * mat->NumCols() + j]);
		}
	}
}

void SSRAlgorithmCUDA::toRRMat(float *resMat, mat_RR *mat){
	for(int i = 0; i < mat->NumRows(); i++){
		for(int j = 0; j < mat->NumCols(); j++){
			(*mat)[i][j] = to_RR(resMat[i * mat->NumCols() + j]);
		}
	}
}

void SSRAlgorithmCUDA::toZZVec(typeV *resVec, vec_ZZ *vec){
	for(int i = 0; i < vec->MaxLength(); i++){
		(*vec)[i] = to_ZZ(resVec[i]);
	}
}

short SSRAlgorithmCUDA::checkForShort(int num){
	if(num > SHRT_MAX){
		Log::error("Entries in Vector/Matrix are too big for type short!");
		exit(-1);
	}
	return short(num);
}

bool SSRAlgorithmCUDA::isZeroVector(vec_ZZ *vec){

	for(int i = 0; i < vec->length(); i++){
		if((*vec)[i]!=0)
			return false;
	}
	return true;
}

void SSRAlgorithmCUDA::AddMinVecsToG(typeV *shortV, unsigned long* normV, int* numAddV, int numAddMax, unsigned long* minNormV, vec_ZZ* minV, int numBlocks, long* numSamples){

	list<unsigned long> sortedNorms;
	list<unsigned long>::iterator it = sortedNorms.begin();

	int i = 0, j = 0, l = 0;

	vec_ZZ v;
	v.SetLength(this->B.NumCols());

	sortedNorms.clear();
	*numAddV = 0;
	*minNormV = ULONG_MAX;
	*numSamples = 0L;

	this->G = this->B;

	/* Sort all valid results into the first numAddMax rows of G */
	for(i = 0; i < numBlocks && this->goalNorm < *minNormV; i++){
		if(normV[i] != 0L){
			(*numSamples) = i+1;
			this->toZZVec(&shortV[i*G.NumCols()], &v);
			/* First smaller vector */
			if(sortedNorms.empty()){
				/* Extend G for the new vector */
				this->G.SetDims(G.NumRows()+1,G.NumCols());
				/* Write the norm of the new vector to a corresponding list */
				sortedNorms.push_back(normV[i]);
				/* Shift all rows to the back */
				for(j = this->G.NumRows()-1; j > 0; j--){
					this->G[j] = this->G[j-1];
				}
				this->G[0] = v;
			}
			else{
				l = 0;
				it = sortedNorms.begin();
				/* Search for the position of the new vector */
				while(it != sortedNorms.end() && *it < normV[i]) {
					l++;
					it++;
				}
				/* Put the new vector to the right position */
				if(l < numAddMax && l < this->G.NumCols()){

					if((int)sortedNorms.size() < numAddMax){
						sortedNorms.insert(it, normV[i]);

						this->G.SetDims(G.NumRows()+1,G.NumCols());

						for(j = this->G.NumRows()-1; j > l; j--){
							this->G[j] = this->G[j-1];
						}
					}
					else{
						*it = normV[i];

						for(j = numAddMax-1; j > l; j--){
							this->G[j] = this->G[j-1];
						}
					}
					this->G[l] = v;
				}
			}
			/* Save smallest vector for logging */
			if(normV[i] < *minNormV){
				*minNormV = normV[i];
				*minV = v;
			}
			(*numAddV)++;

		}
	}
}

void SSRAlgorithmCUDA::RemZeroVecs(){
	int i = 0, j = 0;

	while(isZeroVector(&this->G[i])) i++;
	for(j = 0; j < this->B.NumCols(); j++){
				this->B[j] = this->G[j+i];
	}
	G.SetDims(G.NumRows() - i, G.NumCols());
}
