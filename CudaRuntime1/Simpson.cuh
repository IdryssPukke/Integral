#include "Parameters.cu"

#ifndef __SIMPSON_H__
#define __SIMPSON_H__

#ifdef __cplusplus
extern "C"
{
#endif
	state simpson_method(state a, state b, int steps);
	state simpson_method_CUDA(state a, state b, int steps);
	state simpson_method_CUDA_red(state a, state b, int steps);

#ifdef __cplusplus
}
#endif

#endif /* __SIMPSON_H__ */