#pragma once

#ifdef PLATFORM_LINUX
#include <cuda_runtime.h>
#include <helper_cuda.h>													// includes cuda.h and cuda_runtime_api.h
#include <helper_timer.h>
#include <helper_functions.h>

#else //OSX, earlier CUDA
#include <cuda_runtime.h>
#include <helper_cuda.h>													// includes cuda.h and cuda_runtime_api.h
#include <helper_timer.h>
#include <helper_functions.h>
#endif