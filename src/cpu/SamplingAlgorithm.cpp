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

#include "include/cpu/SamplingAlgorithm.h"
#include <NTL/mat_ZZ.h>

SamplingAlgorithm::SamplingAlgorithm(int startIter, long long int stopIter):ISampleAlgorithm(startIter,stopIter) {

	Log::info("Sampling Algorithm will produce " + Log::toStr(stopIter) + " Samples ...");
}

SamplingAlgorithm::~SamplingAlgorithm() {

}

void SamplingAlgorithm::Sample(vec_ZZ *v, mat_ZZ *B, mat_RR *R, int u){

	vec_RR mu;
	int n, y = 0;

	/* decimal place */
	float decPlace;

	/* temporary vars for estimating decimal place */
	float tempF;
	int tempInt;

	/* temporary vectors for mul and sub */
	vec_ZZ tempZZ;
	vec_RR tempRR;

	/* initialize random seed: */
	srand ( time(NULL) );

	/* get number of columns of given basis */
	n = B->NumCols();

	/* normalize u to Input Basis Dimension */
	/* 1 <= u < n */
	u = u % n;

	/* copy last column vector from the basis */
	*v = (*B)[n-1];

	/* copy last column vector from the Gram-Schmidt coefficient-matrix */
	mu = (*R)[n-1];

	for (int i = n-2; i >= 0; i--){

		if(i < n-u){
			/* search for a y that fits into |mu-y| <= 0.5 */

			/*
			 * There is no need for a random y in general,
			 * because there is only one y that fits.
			 * There is only the case, if the decimal place of
			 * Mu is equal 0.5. In this case y can be two values,
			 * which will estimated randomly.
			 */

			/* filter decimal place */
			tempF = to_float(mu[i]);
			tempInt = (int)tempF;
			decPlace = abs(tempF - tempInt);

			/* estimate y */
			if(decPlace < 0.5){
				y = tempInt;
			}
			else if(decPlace > 0.5){
				if(tempF < 0){
					y = tempInt - 1;
				}
				else{
					y = tempInt + 1;
				}
			}
			else if(decPlace == 0.5){
				if(rand() % 2 == 0){
					y = tempInt;
				}
				else{
					if(tempF < 0){
						y = tempInt - 1;
					}
					else{
						y = tempInt + 1;
					}
				}
			}
		}
		else{

			/* search for a y that fits into |mu-y| <= 1 */

			/* filter decimal place */
			tempF = to_float(mu[i]);
			tempInt = (int)tempF;
			decPlace = abs(tempF - tempInt);

			/* estimate y */
			if(decPlace != 0.0){
				if(rand() % 2 == 0){
					y = tempInt;
				}
				else{
					if(tempF < 0){
						y = tempInt - 1;
					}
					else{
						y = tempInt + 1;
					}
				}
			}
			else{

				switch (rand() % 3) {
					case 0:
						y = tempInt - 1;
						break;
					case 1:
						y = tempInt;
						break;
					case 2:
						y = tempInt + 1;
						break;
					default:
						break;
				}
			}
		}

		/* build new vector v */
		/**
		 * With the following if's, more sample can be produced,
		 * because the value of y is very often -1,0 and 1.
		 */
		if(y==1){
			sub(*v, *v, (*B)[i]);
			sub(mu, mu, (*R)[i]);
		}
		else if(y==-1){
			add(*v, *v, (*B)[i]);
			add(mu, mu, (*R)[i]);
		}
		else if(y!=0){
			mul(tempZZ, y, (*B)[i]);
			sub(*v, *v, tempZZ);

			mul(tempRR, y, (*R)[i]);
			sub(mu, mu, tempRR);
		}
	}
}

int SamplingAlgorithm::getStartIter(){
	return this->startIter;
}

long long int SamplingAlgorithm::getStopIter(){
	return this->stopIter;
}

