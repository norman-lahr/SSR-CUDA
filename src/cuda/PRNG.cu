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

__constant__ unsigned int c1[4]={0xbaa96887, 0x1e17d32c, 0x03bcdc3c, 0x0f33d1b2};
__constant__ unsigned int c2[4]={0x4b0f3b58, 0xe874f0c3, 0x6955c5a6, 0x55a7ca46};

__device__ void psdes(unsigned int *lword, unsigned int *irword)
{
	unsigned int i,ia,ib,iswap,itmph=0,itmpl=0;

	for (i=0;i<4;i++) {

		ia=(iswap=(*irword)) ^ c1[i]; // The bit-rich constants c1 and (below)
									  // c2 guarantee lots of nonlinear mixing.
		itmpl = ia & 0xffff;
		itmph = ia >> 16;
		ib=itmpl*itmpl+ ~(itmph*itmph);
		*irword=(*lword) ^ (((ia = (ib >> 16) |
		((ib & 0xffff) << 16)) ^ c2[i])+itmpl*itmph);
		*lword=iswap;
	}
}

__constant__ unsigned long jflone = 0x3f800000;
__constant__ unsigned long jflmsk = 0x007fffff;

__device__ float rand(int *idum, int *idums)
{
	unsigned int irword,itemp,lword;

	if (*idum < 0) {
		*idums = -(*idum);
		*idum=1;
	}

	irword=(*idum);
	lword=*idums;

	psdes(&lword,&irword);
	itemp=jflone | (jflmsk & irword);
	++(*idum);

	return (*(float *)&itemp)-1.0;
}

__device__ int rand_Discrete(int  *idum, int *idums, int num_discrete_vals) {
	float val;
	val = (float) rand(idum, idums);
	val *= num_discrete_vals;
	val = (int) val;
	return val;
}

#endif /* PRNG_H_ */
