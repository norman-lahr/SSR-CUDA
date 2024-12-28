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

#include "include/cpu/SampleAlgorithm.h"
#include <NTL/mat_ZZ.h>

SampleAlgorithm::SampleAlgorithm(int startIter, long long int stopIter):ISampleAlgorithm(startIter,stopIter) {

	Log::info("Sample Algorithm will produce " + Log::toStr(stopIter) + " Samples ...");
}

SampleAlgorithm::~SampleAlgorithm() {

}

void SampleAlgorithm::Sample(vec_ZZ *v, mat_ZZ *B, mat_RR *R, int u){

	int n = (*B).NumCols(), y = 0, x = u;

	vec_ZZ tempZZ;
	vec_RR mu, tempRR;

	/* Copy last vector in B to v */
	*v = (*B)[n-1];

	/* Copy last vector in R to mu */
	mu = (*R)[n-1];

	for(int j = n-2; j >= 0; j--){
		y = ceil(to_float(mu[j]) - 0.5);

		if( x % 2 == 1){
			if(to_float(mu[j]) - (float)y <= 0.0){
				y -= 1;
			}
			else{
				y += 1;
			}
		}

		x = x >> 1; // x/2

		/* build new vector v */

		/**
		 * With the following if's, more sample can be produced,
		 * because the value of y is very often -1,0 and 1.
		 */
		if(y==1){
			sub(*v, *v, (*B)[j]);
			sub(mu, mu, (*R)[j]);
		}
		else if(y==-1){
			add(*v, *v, (*B)[j]);
			add(mu, mu, (*R)[j]);
		}
		else if(y!=0){
			mul(tempZZ, y, (*B)[j]);
			sub(*v, *v, tempZZ);

			mul(tempRR, y, (*R)[j]);
			sub(mu, mu, tempRR);
		}
	}
}

int SampleAlgorithm::getStartIter(){
	return this->startIter;
}

long long int SampleAlgorithm::getStopIter(){
	return this->stopIter;
}

