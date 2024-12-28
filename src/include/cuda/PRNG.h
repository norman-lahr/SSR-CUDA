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

#ifndef PRNG_H_
#define PRNG_H_

/**
 * Help function for generating random numbers
 * “Pseudo-DES” hashing of the 64-bit word (lword,irword). Both 32-bit
 * arguments are returned hashed on all bits.
 */
__device__ void psdes(unsigned int *lword, unsigned int *irword);

/**
 * Randomizer that produces uniformly distributed float values
 * in range 0..1.
 * \param idum: static variable for random number generation
 * \param idums: static variable for random number generation
 */
__device__ float rand(int *idum, int *idums);

/**
 * Randomizer to generate integer values within the range 0..num_discrete_vals
 */
__device__ int rand_Discrete(int  *idum, int *idums, int num_discrete_vals);

#endif /* PRNG_H_ */
