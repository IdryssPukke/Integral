#include "Parameters.cu"

#ifndef __GAUSS_LEGENDRE_H__
#define __GAUSS_LEGENDRE_H__

#ifdef __cplusplus
extern "C"
{
#endif
	state gauss_legendre(int n, state (*f)(state), state a, state b);

	/* 2D Numerical computation of int(f(x,y),x=a..b,y=c..d) by Gauss-Legendre n-th order high precision quadrature
		[in]n     - quadrature order
		[in]f     - integrand
		[in]data  - pointer on user-defined data which will
					be passed to f every time it called (as third parameter).
		[in][a,b]x[c,d] - interval of integration
	return:
			-computed integral value or -1.0 if n order quadrature is not supported
	*/

	void gauss_legendre_tbl(int n, state* x, state* w, state eps);

#ifdef __cplusplus
}
#endif

#endif /* __GAUSS_LEGENDRE_H__ */