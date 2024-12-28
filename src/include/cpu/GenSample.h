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

#ifndef GENSAMPLE_H_
#define GENSAMPLE_H_

/* Sample interface */
#include "include/cpu/ISampleAlgorithm.h"

/* Log feature */
#include "include/common/Log.h"

/**
 * Class for Sampling Algorithm GenSample.
 */
class GenSample: public ISampleAlgorithm {
public:
	/**
	 * Approach for sampling new vectors from a lattice.
	 *
	 * This Algorithm is originated by the paper
	 * "Shaping Search for Finding Shortest Vectors with High Probability".
	 * But this paper was never published.
	 *
	 * \param startIter Starting point for Sampling
	 * \param n	Dimension of lattice basis
	 * \param j0 Parameter for GenSample
	 * \param k Parameter for GenSample
	 * \param a Parameter for GenSample
	 */
	GenSample(int startIter, int n, int j0, float k, float a);

	/**
	 * Destructor.
	 */
	~GenSample();

	/**
	 * GenSample
	 * \param v sampled vector
	 * \param B lattice basis
	 * \param R Gram-Schmid Coefficient Matrix
	 * \param x Parameter x of the sample algorithm
	 */
	void Sample(vec_ZZ *v, mat_ZZ *B, mat_RR *R, int x);

	/**
	 * Getter for the starting number of the iteration.
	 */
	int getStartIter();

	/**
	 * Getter for the stopping number of the iteration.
	 */
	long long int getStopIter();

private:
	/**
	 * Compute vector c.
	 * \param n Dimension of the vector.
	 * \param j0 Parameter j0
	 * \param k Parameter k
	 * \param a Parameter a
	 */
	vec_ZZ ComputeC(int n, int j0, float k, float a);

	vec_ZZ c;
};

#endif /* GENSAMPLE_H_ */
