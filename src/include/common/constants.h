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

#ifndef CONSTANTS_H_
#define CONSTANTS_H_

//#define DEBUG

/* Implemented Algrotihms for building vector samples */
#define SAMPLING_ALGORITHM		0
#define GENSAMPLE				1
#define SAMPLE_ALGORITHM		2
#define SAMPLE_ALGORITHM_CUDA	3

/* Available reducing algorithms */
#define __BKZ 0
#define __LLL 1

/* Error Codes */
#define U_SMALLER_ZERO					0
#define U_GREATER_DIMENSION				1
#define U_GREATER_ARCHITECTURE			2
#define OUTPUT_FILE_COULD_NOT_OPENED	3
#define ALGORITHM_NOT_SUPPORTED			4

/**
 * The Arguments of the CUDA Kernel
 * are stored in the Shared Memory.
 * Therefore there is not the hole
 * Shared Memory available.
 *
 * Arguments (64bit OS):
 * dimBlock & dimGrid: 16 Byte
 * Pointer:				8 Byte
 * short				2 Byte
 * int:					4 Byte
 * long:				8 Byte
 */
#define AVAILABLE_SHARED_MEM(pointer, shorts, ints, longs) (devProps.sharedMemPerBlock - 16 - (pointer)*sizeof(int*) - (shorts)*sizeof(short) - (ints)*sizeof(int) - (longs)*sizeof(long))
#define AVAILABLE_SHARED_MEM_SAMPLEKERNEL AVAILABLE_SHARED_MEM(3, 0, 0, 0)

#define typeV int

#endif /* CONSTANTS_H_ */
