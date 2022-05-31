#include "Parameters.cu"

#ifndef __SIMPSON38_H__
#define __SIMPSON38_H__

#ifdef __cplusplus
extern "C"
{
#endif
	//state integral_simpson38(state a, state b, state (*f1)(state), int steps);
	state simpson38_method(state a, state b, int steps);
	__global__ void simpson38_method_Kernel(state a, int steps, state step, state* results);
	state simpson38_method_CUDA(state a, state b, int steps);

#ifdef __cplusplus
}
#endif

#endif /* __SIMPSON38_H__ */