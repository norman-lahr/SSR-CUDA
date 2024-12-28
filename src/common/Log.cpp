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

#include "include/common/Log.h"

#define BUFSIZE 21
ofstream Log::outputFile("log.txt", ios_base::app);
bool Log::started = false;
bool Log::verbose = true;
bool Log::lastCR = false;

Log::Log() {

}

Log::~Log() {
	outputFile.close();
}

void Log::onlyToFile(){
	verbose = false;
}

bool Log::isVerbose(){
	return verbose;
}

void Log::print(string msg, bool toFile, bool toDisplay, bool cr){

	char buffer[BUFSIZE];

	/* Get date and time */
	time_t timeSinceEpoch;
	time( &timeSinceEpoch );   // get the time since epoch
	tm *currentTime = localtime( &timeSinceEpoch );  // convert to broken-down local time

	/* Write msg to display and logfile */
	if(strftime( buffer, BUFSIZE, "%d.%m.%Y %H:%M:%S ", currentTime)){
		if(!started){
			started = true;
			outputFile <<
					endl <<
					endl <<
					endl <<
					"****************************************" << endl <<
					"* Starting SSR at " << buffer << " *" << endl <<
					"****************************************" << endl;
		}

		if(toDisplay){
			/* If last call was done with option "cr", the actual line has to be deleted */
			if(lastCR){
				cout << "\r                                                                                                    ";
				fflush(stdout);
				cout << "\r";
				fflush(stdout);
			}

			cout << buffer << " " << msg;

			if(cr){
				cout << "\r";
				fflush(stdout);
			}
			else{
				cout << endl;
			}
			lastCR = cr;
		}
		if(toFile){
			outputFile << buffer << " " << msg << endl;;
		}
	}
}

void Log::info(string msg, bool cr){
	print("<Info> " + msg, true, verbose, cr);
}

void Log::info(string msg, bool toFile, bool toDisplay, bool cr){
	print("<Info> " + msg, toFile, toDisplay && verbose, cr);
}

void Log::debug(string msg){
	print("<Debug> " + msg, true, verbose);
}

void Log::debug(string msg, bool toFile, bool toDisplay){
	print("<Debug> " + msg, toFile, toDisplay && verbose);
}

void Log::error(string msg){
	print("<Error> " + msg, true, true);
}

void Log::error(string msg, bool toFile, bool toDisplay){
	print("<Error> " + msg, toFile, toDisplay);
}

string Log::toStr(char **val, int length){
	string res = "";

	for(int i = 0; i < length; i++){
		res += string(val[i]);
		res += " ";
	}
	return res;
}

