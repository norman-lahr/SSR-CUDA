################################################################################
#
# Build script for project
#
################################################################################

# Add source files here
EXECUTABLE	:= SSR
# Cuda source files (compiled with cudacc)
CUFILES		:= \
cuda/SamplingCUDA.cu

# C/C++ source files (compiled with gcc / c++)
CCFILES		:= common/Log.cpp \
cpu/GenSample.cpp \
cpu/ISampleAlgorithm.cpp \
cpu/SampleAlgorithm.cpp \
cpu/SamplingAlgorithm.cpp \
cpu/SSRAlgorithm.cpp \
cuda/SSRAlgorithmCUDA.cpp \
common/main.cpp

################################################################################
# add Libraries for big numbers
USERLIB := -lntl -lgmp
OPTIMIZING := -msse4 -march=core2

# Compute Capability
CUDACCFLAGS	:= -arch=sm_13

#dbg=1

include common.mk

