#include "Parameters.cu"

#ifndef __SIMPSON38_H__
#define __SIMPSON38_H__

#ifdef __cplusplus
extern "C"
{
#endif
	state simpson38_method(state a, state b, int steps);
	state simpson38_method_CUDA(state a, state b, int steps);
	state simpson38_method_CUDA_red(state a, state b, int steps);

#ifdef __cplusplus
}
#endif

#endif /* __SIMPSON38_H__ */