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

#include "include/cpu/SSRAlgorithm.h"

SSRAlgorithm::SSRAlgorithm(mat_ZZ B, int reducingAlg, int initBlocksize, int inLoopBlocksize, float delta, int initPrecision, int inLoopPrecision, float goalNorm, string filename) {

	this->B = B;
	this->G.SetDims(B.NumRows()+1,B.NumCols());

	this->reducingAlg = reducingAlg;
	this->initBlocksize = initBlocksize;
	this->inLoopBlocksize = inLoopBlocksize;
	this->delta = delta;
	this->initPrecision = initPrecision;
	this->inLoopPrecision = inLoopPrecision;
	this->goalNorm =(unsigned long) (goalNorm * goalNorm);

	this->filename = filename;
}

SSRAlgorithm::~SSRAlgorithm() {

}

mat_ZZ SSRAlgorithm::run(ISampleAlgorithm *pAlgorithm){

	/* loop iterator */
	int x = 0;

	/* Gram-Schmidt coefficient matrix */
	mat_RR R;

	/* Lengths of orthogonal vectors */
	vec_RR b;

	/* Shorter vector*/
	vec_ZZ v,c;

	/* Euclidean Norm of first basis vector */
	unsigned long normB1 = 0L;

	/* Euclidean Norm of vector v */
	unsigned long normV = 1;

	/* found a smaller vector v */
	bool foundSmaller = true;

	/* Times for time measurement */
	struct timeval start, end;
	long mtime, seconds, useconds;
	long int numSamples = 0;

	/* File output stream */
	ofstream outputFile;

	/* Apply Reducing-Algorithm */
	Log::info("Apply first LLL ...");
	NTL::LLL_RR(this->B, delta);
	if(this->reducingAlg != __LLL)
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
	/* Compute Gram-Schmidt decomposition */
	NTL::ComputeGS(this->B, R, b);

	/* Calculate the norm of the first vector in lattice B, before sampling starts */
	euclidNormSqr(&normB1, &B[0]);

	/* Search for a smaller vector */
	Log::info("Start searching for new vectors with SSR algorithm ...");
	while (foundSmaller && (this->goalNorm < normB1)){

		foundSmaller = false;

		Log::info("Norm(B[0]) = " + Log::toStr((long)sqrt(normB1)));

		/* Reset total number of produced samples */
		numSamples = 0;

		/* measure the starting time */
		gettimeofday(&start, NULL);

		/* start reduction */
		for(x = pAlgorithm->getStartIter(); (x < pAlgorithm->getStopIter()) && !foundSmaller; x++){

			pAlgorithm->Sample(&v, &this->B, &R, x);

			/* Calculate the norm of v and check if it is smaller than 99% of normB1 */
			euclidNormSqrGoalNorm(&normV, &v, &normB1, &foundSmaller);

			/* Put out the sampling rate after 1000 samples*/
			if(x % 10000 == 9999){
				gettimeofday(&end, NULL);
				seconds  = end.tv_sec  - start.tv_sec;
				useconds = end.tv_usec - start.tv_usec;

				mtime = seconds*1000 + useconds/1000;
				/* 10000 samples in mtime milliseconds */
				numSamples += 10000000/mtime;

				Log::info("Progress: " + Log::toStr((int)(((float)x/(float)pAlgorithm->getStopIter())*100)) + "%" + " (" + Log::toStr(x+1) + " Samples)" + "\t" +
									"Samplerate: " + Log::toStr(10000000/mtime) + " Samples/sec", false, true, true);

				gettimeofday(&start, NULL);
			}
		}

		/* If the for-loop breaks because of a smaller vector v */
		/* insert v into G */
		if(foundSmaller){
			/* Set v as first columns vector in lattice G */

			for(int i = this->B.NumCols(); i > 0; i--){
				this->G[i] = this->B[i-1];
			}
			this->G[0] = v;

			Log::info("Found smaller vector! \n\t\t\tNorm([new vector]) = " +
					Log::toStr((long)sqrt(normV)) + " (" + Log::toStr((long)(((double)normV/(double)normB1)*100.0)-100L) + "%)" + "\n\t\t\t" +
								"[new vector] = " + Log::toStr(v) + "\n");

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

			/* if the reducing algortihm delete the vector v */
			if(isZeroVector(&this->G[0])){
				for(int i = 0; i < this->B.NumCols(); i++){
					this->B[i] = this->G[i+1];
				}
			}
			else{
				this->B = this->G;
			}

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

	unsigned long norm = 0;
	euclidNormSqr(&norm, &B[0]);
	Log::printNorm((long)sqrt(norm));

	delete pAlgorithm;
	pAlgorithm = NULL;

	return B;
}

void SSRAlgorithm::euclidNormSqrGoalNorm(unsigned long *res, vec_ZZ *vec, unsigned long *goalNorm, bool *foundSmaller){

	ZZ tempRes = to_ZZ(0);

	for(int i = 0; i < vec->length(); i++){
		tempRes += (*vec)[i] * (*vec)[i];

		/* break, if the sum is already greater than 99% of the norm of b[0] */
		if(tempRes > 0.99*(*goalNorm)){
			*foundSmaller = false;

			return;
		}
	}
	if(tempRes <= 0.99* (*goalNorm)){
		*foundSmaller = true;
		*res = to_ulong(tempRes);
	}
}

void SSRAlgorithm::euclidNormSqr(unsigned long *res, vec_ZZ *vec){

	ZZ tempRes = to_ZZ(0);

	for(int i = 0; i < vec->length(); i++){
		tempRes += (*vec)[i] * (*vec)[i];
	}

	if(tempRes > to_ZZ(ULONG_MAX)){
		Log::error("euclidNormSqr: Norm of vector is too big for type unsigned long!\nSearching canceled");
		exit(-1);
	}
	*res = to_ulong(tempRes);
}

bool SSRAlgorithm::isZeroVector(vec_ZZ *vec){

	for(int i = 0; i < vec->length(); i++){
		if((*vec)[i]!=0)
			return false;
	}
	return true;
}
