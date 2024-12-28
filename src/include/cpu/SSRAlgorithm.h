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

#ifndef SSRALGORITHM_H_
#define SSRALGORITHM_H_

/* for random numbers */
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <cstdio>
#include <cmath>
#include <climits>

/* for ofstream and ifstream */
#include <fstream>

#include "include/cpu/ISampleAlgorithm.h"
#include "include/cpu/SamplingAlgorithm.h"
#include "include/cpu/GenSample.h"
#include "include/cpu/SampleAlgorithm.h"

/* NTL library for big integers */
#include <NTL/mat_ZZ.h>
#include <NTL/LLL.h>
#include <NTL/ZZ.h>
NTL_CLIENT
/* Common constants */
#include "include/common/constants.h"
/* Logging feature */
#include "include/common/Log.h"

/**
 * Class for main part of SSR algorithm on CPU
 */
class SSRAlgorithm {
public:
	/**
	 * Constructor for the Simple Sampling Algorithm.
	 *
	 * The algorithm is from the paper
	 * "Practical Lattice Basis Sampling Reduction"
	 * written by Johannes Buchmann and Christoph Ludwig.
	 *
	 * \param B Lattice Basis
	 * \param uMax Samplespace
	 * \param reducingAlg Algorithm-index for lattice reduction
	 * \param blocksize Blocksize for BKZ algorithm (optional)
	 */
	SSRAlgorithm(mat_ZZ B, int reducingAlg, int initBlocksize, int inLoopBlocksize, float delta, int initPrecision, int inLoopPrecision, float goalNorm, string filename);
	virtual ~SSRAlgorithm();

	/**
	 *  Runs the Simple Sampling Reduction Algorithm.
	 *  \param Algorithm object
	 */
	mat_ZZ run(ISampleAlgorithm *pAlgorithm);

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
	 * Calculates the euclidean norm of a vector, and breaks up if
	 * the temporary sum is already grater than 99% of the norm of b[0]
	 * \param res Result of euclidean norm calculation.
	 * \param vec Vector, whose norm will be calculated.
	 * \param goalNorm Goal norm of the current session.
	 * \param foundSmaller Indicates, if the calculated norm is smaller than the goal norm.
	 */
	void euclidNormSqrGoalNorm(unsigned long *res, vec_ZZ *vec, unsigned long *goalNorm, bool *foundSmaller);

	vec_ZZ ComputeC(int n, int j0, float k, float a);
	/**
	 * Checks if all vector entries are zero.
	 * \param Vector to test
	 */
	bool isZeroVector(vec_ZZ *vec);
};

#endif /* SSRALGORITHM_H_ */
