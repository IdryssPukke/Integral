#include "Parameters.cu"

#ifndef __RECTANGLE_H__
#define __RECTANGLE_H__

#ifdef __cplusplus
extern "C"
{
#endif
	state rectangle_method(state a, state b, int n);
	state rectangle_method_CUDA(state a, state b, int steps);
	state rectangle_method_CUDA_red(state a, state b, int steps);
	state rectangle_method_CUDA_red2(state a, state b, int steps);

#ifdef __cplusplus
}
#endif

#endif /* __RECTANGLE_H__ */