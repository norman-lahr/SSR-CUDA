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

#ifndef MAIN_H_
#define MAIN_H_

/* Standard stream */
#include <iostream>
/* for ofstream and ifstream */
#include <fstream>

#include <cstdlib>
#include <string>
#include <sstream> //TODO streams alle nötig?

/* NTL library for big integers */
#include <NTL/mat_ZZ.h>
NTL_CLIENT

/* For reading options from command line */
#include <getopt.h>

/* Main part of both SSR variants */
#include "include/cpu/SSRAlgorithm.h"
#include "include/cuda/SSRAlgorithmCUDA.h"

/* common constants */
#include "include/common/constants.h"
/* Log feature */
#include "include/common/Log.h"

using namespace std;

/**
 * Transforms all characters of a string to lower case.
 * \param str String to convert.
 */
string tolowerStr(string str);

/**
 * Prints the usage of the program.
 */
void usage();

/**
 * Prints help.
 */
void help();
#endif /* MAIN_H_ */
