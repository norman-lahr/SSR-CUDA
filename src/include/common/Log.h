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

#ifndef LOG_H_
#define LOG_H_

/* Standard stream */
#include <iostream>
/* for fstream, ifstream and stringstream*/
#include <fstream>
#include <sstream>
#include <string>
/* for time and date */
#include <ctime>
#include <stdlib.h>
/* for norm printing */
#include <NTL/ZZ.h>
#include <NTL/vec_ZZ.h>
NTL_CLIENT

using namespace std;

/**
 * Class for Logging.
 */
class Log {
public:
	/**
	 * Constructor.
	 */
	Log();

	/**
	 * Destructor
	 */
	virtual ~Log();

	/**
	 * The output of the display is only the square norm.
	 * All other information are printed in log file.
	 */
	static void onlyToFile();

	/**
	 * Tells, if logger is in verbose mode.
	 */
	static bool isVerbose();

	/**
	 * If logger is in verbose mode, this function prints an info line with a string message.
	 * \param msg Message to print
	 */
	static void info(string msg, bool cr = false);

	/**
	 * If logger is in verbose mode, this function prints an info line with a string message.
	 * \param msg Message to print
	 * \param toFile Should the message be written to log file?
	 * \param toDisplay Should the message be written to display?
	 */
	static void info(string msg, bool toFile, bool toDisplay, bool cr = false);

	/**
	 * If logger is in verbose mode, this function prints an debug line with a string message.
	 * \param msg Message to print
	 */
	static void debug(string msg);

	/**
	 * If logger is in verbose mode, this function prints a debug line with a string message.
	 * \param msg Message to print
	 * \param toFile Should the message be written to log file?
	 * \param toDisplay Should the message be written to display?
	 */
	static void debug(string msg, bool toFile, bool toDisplay);

	/**
	 * This function prints an error line with a string message.
	 * \param msg Message to print
	 */
	static void error(string msg);

	/**
	 * This function prints an error line with a string message.
	 * \param msg Message to print
	 * \param toFile Should the message be written to log file?
	 * \param toDisplay Should the message be written to display?
	 */
	static void error(string msg, bool toFile, bool toDisplay);

	/**
	 * Prints a vector, if logger is in verbose mode. Implemented in *.h, because
	 * "export" is not supported by gcc-4.3
	 * \param vec Vector to print
	 * \param len Length of vector
	 * \param name Name of the vector
	 */
	template <typename T> static void printVector(T vec, int len, string name){
		stringstream str;

		str << name << " = [";
		for(int i=0; i < len; i++){
		   str << " " << vec[i];
		}
		str << " ]\n";
		print(str.str(),true, verbose);
	}

	/**
	 * Prints a vector, if logger is in verbose mode. Implemented in *.h, because
	 * "export" is not supported by gcc-4.3
	 * \param vec Vector to print
	 * \param len Length of vector
	 * \param toFile Should the message be written to log file?
	 * \param toDisplay Should the message be written to display?
	 * \
	 */
	template <typename T> static void printVector(T vec, int len, string name, bool toFile, bool toDisplay){
		stringstream str;

		str << name << ": ";
		for(int i=0; i < len; i++){
		   str << " " << vec[i];
		}
		str << "\n";
		print(str.str(), toFile, toDisplay && verbose);
	}

	/**
	 * Prints square norm, if verbose is false. This is only for
	 * analysing.
	 * \param norm Final norm
	 */
	template <typename T> static void printNorm(T norm){
		print("<Final Norm> " + Log::toStr(norm), true, verbose);
		if(!verbose)
			cout << norm << endl;
	}

	/**
	 * Converts to string representation.
	 * \param val Value to convert.
	 */
	template <typename T> static string toStr(T val){
		stringstream s;

		s << val;
		return s.str();
	}

	/**
	 * Converts the program call to string.
	 * \param val typically *argv[]
	 * \param length typically argc
	 */
	static string toStr(char **val, int length);

private:
	/* Log file */
	static ofstream outputFile;

	/* Stores, whether the logger is already started */
	static bool started;

	/* Stores, whether the logger is in verbose mode */
	static bool verbose;

	/* Stores the last status of the carriage return variable */
	static bool lastCR;

	/**
	 * General print function. Prints a message to display and to a log file.
	 * \param msg Message to print
	 * \param toFile Should the message be written to log file?
	 * \param toDisplay Should the message be written to display?
	 * \param cr Do a carriage return for displaying continuous data.
	 */
	static void print(string msg, bool toFile, bool toDisplay, bool cr = false);
};

#endif /* LOG_H_ */
