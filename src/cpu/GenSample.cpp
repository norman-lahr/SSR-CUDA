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

#include "include/cpu/GenSample.h"
#include <NTL/mat_ZZ.h>

GenSample::GenSample(int startIter, int n, int j0, float k, float a):ISampleAlgorithm(startIter) {

	/* Compute vector c */
	this->c = ComputeC(n, j0, k, a);

	/* N = |W_k,a,j0,B| , number of samples*/
	this->stopIter = 1;
	for(int j = 0; j < j0; j++){
		this->stopIter *= to_int(c[j]);
	}

	Log::info("GenSample will produce " + Log::toStr(this->stopIter) + " Samples ...");
}

GenSample::~GenSample() {

}

void GenSample::Sample(vec_ZZ *v, mat_ZZ *B, mat_RR *R, int xIn){

	int n = B->NumRows(),
			z, y, x;

	vec_RR v_, tempRR;
	vec_ZZ tempZZ;

	for(int j = n-1; j >= 0; j--){
		if(this->c[j] > 0){
			x = xIn / to_int(this->c[j]);
			z = xIn % to_int(this->c[j]);

			if((j == n-1) || (this->c[j+1] == 0)){
				mul(*v, x+1, (*B)[j]);
				mul(v_, x+1, (*R)[j]);
			}
			else{
				/* z = (-1)^z * z/2; */
				if(z % 2 == 0){
					z = z >> 1;		/* z/2 */
				}
				else{
					z = -(z >> 1);
				}

				y = ceil(to_float(v_[j]) - 0.5); // round up

				if(v_[j]-y <= 0){
					y -= z;
				}
				else{
					y += z;
				}

				/* build new vector v */
				/**
				 * With the following if's, more sample can be produced,
				 * because the value of y is very often -1,0 and 1.
				 */
				if(y!=0){
					mul(tempZZ, y, (*B)[j]);
					sub(*v, *v, tempZZ);

					mul(tempRR, y, (*R)[j]);
					sub(v_, v_, tempRR);
				}
			}
		}
	}
}

int GenSample::getStartIter(){
	return this->startIter;
}

long long int GenSample::getStopIter(){
	return this->stopIter;
}

vec_ZZ GenSample::ComputeC(int n, int j0, float k, float a){

	vec_ZZ c;
	c.SetLength(n);

	for(int j = 0; j < j0-1; j++){
		c[j] = ceil(2 * k * pow(a, n-j)); // round up
	}

	c[j0-1] = k * pow(a, n-(j0-1));

	for(int j = j0; j < n; j++){
		c[j] = 0;
	}

	return c;
}
