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

#include "include/common/main.h"

/**
 * Input data format:
 * [[1 2 3]
 * [4 5 6]
 * [7 8 9]]
 */

int main(int argc, char *argv[]){

	/* Stream for file input */
	ifstream	inputFile;

	string outputFilename = "";

	/* Input matrix */
	mat_ZZ mBasis;

	/* Help variable for reading options */
	int optc = 0;

	/* Initialize parameters */
	int uMax = 24,
		reducingAlg = 0,
		initBlocksize = 10,
		inLoopBlocksize = -1,
		initPrecision = 3,
		inLoopPrecision = -1,
		samplingAlg = SAMPLE_ALGORITHM,
		j0 = 0,
		useDeviceID = 0,
		numBlocks = 65535,
		numAddVMax = 65535,
		errorCode = -1;

	float delta = 0.99,
			k = 1.5,
			a = 0.95,
			goalNorm = 0.0;

	bool cudaEnabled = false,
			verbose = false;

	/* Log program call */
	Log::info(Log::toStr(argv, argc), true, false);

	/* Read options from command line */
	opterr = 0;

	while((optc = getopt(argc, argv, "m:o:u:r:d:b:B:g:s:j:k:a:c:p:P:n:vh")) != -1){

	  switch(optc){
		case 'm':
			numAddVMax = atoi(optarg);
			break;
		case 'o':
			outputFilename = string(optarg);
			break;
	    case 'u':
	    	uMax = atoi(optarg);
	    	break;
	    case 'r':
	    	if((tolowerStr(string(optarg)).compare("bkz")) == 0)
	    		reducingAlg = __BKZ;
	    	else if ((tolowerStr(string(optarg)).compare("lll")) == 0)
	    		reducingAlg = __LLL;
			break;
	    case 'd':
			delta = atof(optarg);
			break;
	    case 'b':
	    	if(reducingAlg == __BKZ)
	    		initBlocksize = atoi(optarg);
	    	break;
	    case 'B':
	    	if(reducingAlg == __BKZ)
	    		inLoopBlocksize = atoi(optarg);
	    	break;
	    case 'g':
    		goalNorm = atof(optarg);
	    	break;
	    case 's':
	    	if((tolowerStr(string(optarg)).compare("sa")) == 0)
	    		samplingAlg = SAMPLING_ALGORITHM;
	    	else if ((tolowerStr(string(optarg)).compare("gensample")) == 0)
	    		samplingAlg = GENSAMPLE;
	    	else if ((tolowerStr(string(optarg)).compare("sample")) == 0)
	    		    		samplingAlg = SAMPLE_ALGORITHM;
	    	break;
	    case 'j':
	    	if(samplingAlg == GENSAMPLE)
	    		j0 = atoi(optarg);
	    	break;
	    case 'k':
	    	if(samplingAlg == GENSAMPLE)
	    		k = atof(optarg);
	    	break;
	    case 'a':
	    	if(samplingAlg == GENSAMPLE)
	    		a = atof(optarg);
	    	break;
	    case 'c':
	    	if(optarg != NULL)
	    		useDeviceID = atoi(optarg);
	    	else
	    		useDeviceID = 0;

	    	cudaEnabled = true;
	    	break;
	    case 'p':
	    	initPrecision = atoi(optarg);
	    	if(initPrecision > 4 || initPrecision < 1){
	    		usage(); return -1;
	    	}
	    	break;
	    case 'P':
	    	inLoopPrecision = atoi(optarg);
	    	if(inLoopPrecision > 4 || inLoopPrecision < 1){
	    		usage(); return -1;
	    	}
	    	break;
	    case 'v':
	    	verbose = true;
	    	break;
	    case 'n':
	    	numBlocks = atoi(optarg);
	    	break;
	    case 'h':
	    	help();
	    	return 0;
	    default:
	    	usage(); return -1;
	  }
	}

	/* If the in loop parameters are not set, they get the values of the
	 * first parameters */
	if(inLoopBlocksize == -1)
		inLoopBlocksize = initBlocksize;
	if(inLoopPrecision == -1)
		inLoopPrecision = initPrecision;

	/* There must be a filename as argument */
	if(optind == argc - 1){
		/* open input file */
		inputFile.open(argv[argc - 1], ios_base::in);

		/* detect errors */
		if (!inputFile){
		Log::error("File not found!");
		usage();
		/* return with error */
		return -1;
		}
	  }else{
		  Log::error("There are wrong arguments!");
	    usage();
	    return -1;
	  }

	/* Read out file */
	inputFile >> mBasis;
	inputFile.close();

	/* Set default filename for output file */
	if(outputFilename.compare("") == 0){
		ostringstream temp;
		string argString = string(argv[argc - 1]);
		argString.erase(argString.begin() + (int)argString.rfind("."), argString.end());
		temp << argString << "-u" << uMax << "b" << initBlocksize << "B" << inLoopBlocksize << ".reduced";
		outputFilename = temp.str();
	}

	/* Set display mode */
	if(!verbose)
		Log::onlyToFile();

	/* Error detection */
	if(uMax >= mBasis.NumCols())
		errorCode = U_GREATER_DIMENSION;
	if(uMax >= sizeof(long)*8) //sizeof returns unsigned value
		errorCode = U_GREATER_ARCHITECTURE;
	if(uMax <= 0)
		errorCode = U_SMALLER_ZERO;

	/* check uMax */
	if (errorCode == -1){

		if(!cudaEnabled){

			SSRAlgorithm* ssrToken = new SSRAlgorithm(mBasis, reducingAlg, initBlocksize, inLoopBlocksize, delta, initPrecision, inLoopPrecision, goalNorm, outputFilename);

			/* Switch between the implemented sample algorithms */
			switch (samplingAlg) {
				case SAMPLING_ALGORITHM:
					mBasis = ssrToken->run(new SamplingAlgorithm::SamplingAlgorithm(1, (long int)1 << uMax));
					break;
				case GENSAMPLE:
					if(j0 <= 0 || j0 > mBasis.NumRows()) j0 = mBasis.NumRows();
					mBasis = ssrToken->run(new GenSample::GenSample(0,mBasis.NumRows(), j0, k , a));
					break;
				case SAMPLE_ALGORITHM:
					mBasis = ssrToken->run(new SampleAlgorithm::SampleAlgorithm(0, ((long int)1 << (uMax-1))-1));
					break;
				default:
					break;
			}

			delete ssrToken;
		}

		else{

			SSRAlgorithmCUDA* ssrToken = new SSRAlgorithmCUDA(mBasis, reducingAlg, initBlocksize, inLoopBlocksize, delta, initPrecision, inLoopPrecision, goalNorm, outputFilename);

			switch (samplingAlg) {
				case SAMPLING_ALGORITHM:
					mBasis = ssrToken->run(samplingAlg, 1, (unsigned long)1 << uMax, useDeviceID, numBlocks, numAddVMax);
					break;
				case GENSAMPLE:
					errorCode = ALGORITHM_NOT_SUPPORTED;
					break;
				case SAMPLE_ALGORITHM:
					mBasis = ssrToken->run(samplingAlg, 0, ((unsigned long)1 << (uMax-1))-1, useDeviceID, numBlocks, numAddVMax);
					break;
				default:
					break;
			}

			delete ssrToken;
		}

//		/* Save results */
//		outputFile.open(outputFilename.c_str());
//
//		if (!outputFile){
//			errorCode = OUTPUT_FILE_COULD_NOT_OPENED;
//		}
//		else{
//			outputFile << mBasis;
//			outputFile.close();
//		}
	}

	/* Print errors */
	switch (errorCode) {
		case U_SMALLER_ZERO:
			Log::error("Search Space Bound u is smaller than 0. Please choose a value s.t. u > 0.");
			return -1;
			break;
		case U_GREATER_DIMENSION:
			Log::error("Search Space Bound u is greater than the dimension n. Please choose a value s.t. 1 <= u < n.");
			return -1;
			break;
		case U_GREATER_ARCHITECTURE:
			Log::error("Search Space Bound u is greater than the instruction set architecture. Please choose a value, smaller than " + Log::toStr(sizeof(long)*8) + ".");
			return -1;
			break;
		case ALGORITHM_NOT_SUPPORTED:
			Log::error("Sorry, this algorithm is not supported for CUDA yet.");
			return -1;
			break;
		default:
			Log::info("Finish");
			break;
		}

	return 0;
}

string tolowerStr(string str){

	 locale loc;
	 string res;

	  for (size_t i=0; i<str.length(); ++i)
	    res += tolower(str[i],loc);

	  return res;
}

void usage(){
  cout << "Usage: SSR [OPTIONS] [FILE]" << endl;
  cout << "Use option -h for help." << endl;
}

void help(){
  cout << "Usage: SSR [OPTIONS] [FILE]" << endl << endl;
  cout << "List of options:" << endl;
  cout << "\t-c\t<optional DeviceID>\n\t\tEnables CUDA for parallel computation\n\t\twith the selected DeviceID \n\t\t(default: 0)"<<endl;
  cout << "\t-o\t<filename>\n\t\tFilename for saving results" <<endl;
  cout << "\t-n\t<Number of CUDA Blocks> \n\t\t(default: 65535)" << endl;
  cout << "\t-m\t<Add maximal X smaller vectors v> \n\t\t(default: 65535)" << endl;
  cout << "\t-u\t<Sample Space Bound>\n\t\t1 <= u < dimension \n\t\t(default: 24)" << endl;
  cout << "\t-r\t<Reducing Algorithm>\n\t\t[BKZ|LLL] \n\t\t(default: BKZ)"<<endl;
  cout << "\t-d\t<delta>\n\t\t(default: 0.99)"<<endl;
  cout << "\t-b\t<blocksize>\n\t\tBlocksize for the initial BKZ \n\t\t(default: 10)"<<endl;
  cout << "\t-B\t<optional blocksize(in loop)>\n\t\tBlocksize for BKZ in sample loop \n\t\t(default: value of -b)"<<endl;
  cout << "\t-g\t<goal norm>\n\t\tAbort criterion" << endl;
  cout << "\t-p\t<precision(first)>\n\t\tPrecision of the initial BKZ or LLL \n\t\t(default: 3):\n\t\t1: double (FP)\n\t\t2: quasi quadruple precision (QP)"
 		   << "\n\t\t3: extended exponent doubles (XD)\n\t\t4: arbitrary precision floating point (RR)" << endl;
  cout << "\t-P\t<optional precision(in loop)>\n\t\tPrecision of the BKZ or LLL in the sample loop \n\t\t(default: value of -p):\n\t\t1: double (FP)\n\t\t2: quasi quadruple precision (QP)"
 		   << "\n\t\t3: extended exponent doubles (XD)\n\t\t4: arbitrary precision floating point (RR)" << endl;
  cout << "\t-s\t<Sampling Algorithm> \n\t\t[SA|GenSample|Sample] \n\t\t(default: Sample)"<<endl;
  cout << "\t-j\t<j0>\n\t\tParameter for GenSample, smaller than Dimension n of given matrix \n\t\t(default: n)"<<endl;
  cout << "\t-k\t<k> \n\t\tParameter for GenSample, [1.0 .. 2.0]\n\t\t(default: 1.5)"<<endl;
  cout << "\t-a\t<a>\n\t\tParameter for GenSample, [0.8800 .. 0.9990]\n\t\t(default:0.9500)"<<endl;
  cout << "\t-v\tverbose" << endl;
}
