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

#ifndef SAMPLINGCUDA_TEXTURES_H_
#define SAMPLINGCUDA_TEXTURES_H_

/* Storing matrice B and R in texture memory */
texture<short, 1, cudaReadModeElementType> texRefB;
texture<float, 1, cudaReadModeElementType> texRefR;

__constant__ int const_Dim;
__constant__ int const_SamplesPerBlock;
__constant__ unsigned long int const_xOffset;
__constant__ unsigned long const_normB1;
__constant__ bool const_firstTime;
__constant__ int const_numAddMax;

#endif /* SAMPLINGCUDA_TEXTURES_H_ */
