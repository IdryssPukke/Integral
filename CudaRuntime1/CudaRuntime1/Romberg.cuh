#include "Parameters.cu"

#ifndef __ROMBERG_H__
#define __ROMBERG_H__

#ifdef __cplusplus
extern "C"
{
#endif
	state romberg_method(state a, state b,  int N);
	state romberg_method_CUDA_1(state a, state b, int N);
	state romberg_method_CUDA_2(state a, state b, int N);
#ifdef __cplusplus
}
#endif

#endif /* __ROMBERG_H__ */