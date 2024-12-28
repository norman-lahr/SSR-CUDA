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

#ifndef SAMPLINGALGORITHM_H_
#define SAMPLINGALGORITHM_H_

/* Sample interface */
#include "include/cpu/ISampleAlgorithm.h"
/* Logging feature */
#include "include/common/Log.h"

#include <cmath>

/**
 * Class for Sampling Algorithm.
 */
class SamplingAlgorithm: public ISampleAlgorithm {
public:
	/**
	 * Constructor for Sampling Algorithm.
	 *
	 *  This Algorithm is from the paper
	 * "Lattice Reduction by Random Sampling and Birthday Methods"
	 * written by Claus Peter Schnorr.
	 *
	 * \param startIter Starting point of sampling.
	 * \param stopIter Stopping point of sampling.
	 */
	SamplingAlgorithm(int startIter, long long int stopIter);

	~SamplingAlgorithm();

	/**
	 * Sampling function.
	 * \param v Vector v
	 * \param B Lattice Basis B
	 * \param R Gram-Schmidt coefficient-matrix
	 * \param u Iterator resp. Search Space
	 */
	void Sample(vec_ZZ *v, mat_ZZ *B, mat_RR *R, int u);

	/**
	 * Getter for startIter
	 */
	int getStartIter();

	/**
	 * Getter for stopIter
	 */
	long long int getStopIter();
};

#endif /* SAMPLINGALGORITHM_H_ */
