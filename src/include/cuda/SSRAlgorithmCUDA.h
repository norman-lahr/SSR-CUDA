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

#ifndef SSRALGORITHMCUDA_H_
#define SSRALGORITHMCUDA_H_

/**
 * Class for Parallel Sampling Algorithm using CUDA Framework.
 * It calls a Kernel that runs on Nvidia's Graphics Cards.
 */
/* includes, system */
#include <iostream>
#include <stdlib.h>
#include <climits>
#include <sys/time.h>

#include <list>
#include <vector>

/* for ofstream and ifstream */
#include <fstream>

/* NTL library for big integers */
#include <NTL/mat_ZZ.h>
#include <NTL/LLL.h>
#include <NTL/ZZ.h>
NTL_CLIENT

#include "include/common/constants.h"
#include "include/common/Log.h"

/* includes, project */
#include <cutil_inline.h>

/**
 * Class for main part of SSR algorithm on CUDA
 */
class SSRAlgorithmCUDA {
public:
	/**
	 * Constructor for SSR algorithm on CUDA.
	 * \param B lattice basis
	 * \param reducingAlg reducing algorithm
	 * \param initBlocksize initial blocksize for BKZ
	 * \param inLoopBlocksize blocksize for BKZ in the loop
	 * \param delta Parameter delta
	 * \param initPrecision initial precision for BKZ and LLL
	 * \param inLoopPrecision Precision for BKZ and LLL in the loop
	 * \param goalNorm Goal norm
	 */
	SSRAlgorithmCUDA(mat_ZZ B, int reducingAlg, int initBlocksize, int inLoopBlocksize, float delta, int initPrecision, int inLoopPrecision, float goalNorm, string filename);
	virtual ~SSRAlgorithmCUDA();

	/**
	 * Runs the Simple Sampling Reduction Algorithm.
	 * \param SampleAlgorithm type of sample algorithm
	 * \param startIter starting number of iteration
	 * \param stopIter stopping number of iteration
	 * \param deviceID CUDA Device Identifier
	 * \param numBlocks number of CUDA blocks
	 * \param numAddMax maximum number of vectors, which were add to the lattice
	 */
	mat_ZZ run(int SampleAlgorithm, int startIter, unsigned long int stopIter, int deviceID, int numBlocks, int numAddVMax);

private:
	/* internal generator matrix */
	mat_ZZ B, G;

	int initBlocksize;
	int inLoopBlocksize;
	float delta;
	int initPrecision;
	int inLoopPrecision;
	int reducingAlg;
	unsigned long goalNorm;

	string filename;

	/**
	 * Calculates the euclidean norm of a vector.
	 * \param res Result of euclidean norm calculation.
	 * \param vec Vector, whose norm will be calculated.
	 */
	void euclidNormSqr(unsigned long *res, vec_ZZ *vec);
	/**
	 * Checks if all vector entries are zero.
	 * \param vec Vector to test
	 * \return True, if vec contains only zeroes. False, else.
	 */
	bool isZeroVector(vec_ZZ *vec);
	/**
	 * Transform an NTL vec_ZZ to a non-NTL array
	 * \param vec input vector
	 * \param resVec result
	 */
	void toShortVec(typeV *resVec, vec_ZZ *vec);
	/**
	 * Transform an NTL mat_ZZ to a non-NTL matrix
	 * \param resMat result
	 * \param mat input matrix
	 */
	void toShortMat(short *resMat, mat_ZZ *mat);
	/**
	 * Transform an NTL mat_RR to an float matrix
	 * \param resMat result
	 * \param mat input vector
	 */
	void toFloatMat(float *resMat, mat_RR *mat);
	/**
	 * Transforms a short matrix to an NTL mat_ZZ.
	 * Both matrices have to be exists before.
	 * \param resMat result
	 * \param mat input matrix
	 */
	void toZZMat(short *resMat, mat_ZZ *mat);
	/**
	 * Transforms a float matrix to an NTL mat_RR.
	 * Both matrices have to be exists before.
	 * \param resMat result
	 * \param mat input matrix
	 */
	void toRRMat(float *resMat, mat_RR *mat);
	/**
	 * Transforms a short vector to an NTL vec_ZZ.
	 * Both vectors has to be exists before.
	 * \param resVec result
	 * \param vec input vector
	 */
	void toZZVec(typeV *resVec, vec_ZZ *vec);
	/**
	 * Checks if a number fits in an short variable
	 * \param num integer to test
	 */
	short checkForShort(int num);
	/**
	 * Adds the numAddMax smallest vectors to the front of G
	 * \param shortV Vectors v
	 * \param normV Norm of v
	 * \param numAddV Number of v's to add
	 * \param numAddMax Maximum number of v's to add
	 * \param minNormV Smallest norm
	 * \param minV Smallest v
	 * \param numBlocks Number of CUDA blocks
	 * \param numSamples Number of produced samples
	 */
	void AddMinVecsToG(typeV *shortV, unsigned long* normV, int* numAddV, int numAddMax, unsigned long* minNormV, vec_ZZ* minV, int numBlocks, long* numSamples);
	/**
	 * After LLL/BKZ reduction, there are zero vectors in the front of G.
	 * This method adds the non zero vectors of G to B.
	 */
	void RemZeroVecs();
};

#endif /* SSRALGORITHMCUDA_H_ */
