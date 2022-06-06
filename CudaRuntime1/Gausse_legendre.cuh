#include "Parameters.cu"

#ifndef __GAUSS_LEGENDRE_H__
#define __GAUSS_LEGENDRE_H__

#ifdef __cplusplus
extern "C"
{
#endif
	state gauss_legendre_method(state a, state b, int n);
	void gauss_legendre_tbl(int n, state* x, state* w, state eps);
	state gauss_legendre_method_CUDA(state a, state b, int n);
	state gauss_legendre_method_CUDA_red(state a, state b, int n);

#ifdef __cplusplus
}
#endif

#endif /* __GAUSS_LEGENDRE_H__ */