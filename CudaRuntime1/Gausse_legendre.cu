#define _USE_MATH_DEFINES
#include <cmath>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#include <cuda_runtime.h>
#include <nvfunctional>
#include "device_launch_parameters.h"

#include "Gausse_legendre.cuh"
#include "Parameters.cu"

extern __device__ __host__  state fun3(state x) {
	//return sin(x);
	return pow(exp(1.0), x);
}

/*
Gauss-Legendre n-points quadrature, exact for polynomial of degree <=2n-1

1. n - even:

int(f(t),t=a..b) = A*sum(w[i]*f(A*x[i]+B),i=0..n-1) = A * sum( w[k] * [f(B + A * x[k]) + f(B - A * x[k])], k=0..n/2-1)
	A = (b-a)/2,
	B = (a+b)/2

2. n - odd:

int(f(t),t=a..b) = A*sum(w[i]*f(A*x[i]+B),i=0..n-1)
		 = A*w[0]*f(B)+A*sum(w[k]*[f(B+A*x[k])+f(B-A*x[k])],k=1..(n-1)/2)
	A = (b-a)/2,
	B = (a+b)/2
*/

state gauss_legendre_method(state a, state b, int n)
{
	state* x = NULL;
	state* w = NULL;
	state A, B, Ax, s;
	int i, m;

	m = (n + 1) >> 1;

	/* Generate new if non-predefined table is required */
	/* with precision of 1e-10 */

	x = (state*)malloc(m * sizeof(state));
	w = (state*)malloc(m * sizeof(state));

	gauss_legendre_tbl(n, x, w, 1e-10);

	/* Next part*/
	A = 0.5 * (b - a);
	B = 0.5 * (b + a);

	if (n & 1) /* n - odd */
	{
		s = w[0] * (fun3(B));
		for (i = 1; i < m; i++)
		{
			Ax = A * x[i];
			s += w[i] * (fun3(B + Ax) + fun3(B - Ax));
		}

	}
	else { /* n - even */

		s = 0.0;
		for (i = 0; i < m; i++)
		{
			Ax = A * x[i];
			s += w[i] * (fun3(B + Ax) + fun3(B - Ax));
		}
	}

	free(x);
	free(w);
	
	return A * s;
}


/* Computing of abscissas and weights for Gauss-Legendre quadrature for any(reasonable) order n
	[in] n   - order of quadrature
	[in] eps - required precision (must be eps>=macheps(state), usually eps = 1e-10 is ok)
	[out]x   - abscisass, size = (n+1)>>1
	[out]w   - weights, size = (n+1)>>1
*/

/* Look up table for fast calculation of Legendre polynomial for n<1024 */
/* ltbl[k] = 1.0 - 1.0/(state)k; */

void gauss_legendre_tbl(int n, state* x, state* w, state eps)
{
	state x0, x1, dx;	/* Abscissas */
	state w0, w1, dw;	/* Weights */
	state P0, P_1, P_2;	/* Legendre polynomial values */
	state dpdx;			/* Legendre polynomial derivative */
	int i, j, k, m;			/* Iterators */
	state t0, t1, t2, t3;

	m = (n + 1) >> 1;

	t0 = (1.0 - (1.0 - 1.0 / (state)n) / (8.0 * (state)n * (state)n));
	t1 = 1.0 / (4.0 * (state)n + 2.0);

	for (i = 1; i <= m; i++)
	{
		/* Find i-th root of Legendre polynomial */

		/* Initial guess */
		x0 = cos(M_PI * (state)((i << 2) - 1) * t1) * t0;

		/* Newton iterations, at least one */
		j = 0;
		dx = dw = DBL_MAX;
		do
		{
			/* Compute Legendre polynomial value at x0 */
			P_1 = 1.0;
			P0 = x0;

			/* Simple, not optimized version */
			for (k = 2; k <= n; k++)
			{
				P_2 = P_1;
				P_1 = P0;
				t2 = x0 * P_1;
				t3 = (state)(k - 1) / (state)k;

				P0 = t2 + t3 * (t2 - P_2);
			}

			/* Compute Legendre polynomial derivative at x0 */
			dpdx = ((x0 * P0 - P_1) * (state)n) / (x0 * x0 - 1.0);

			/* Newton step */
			x1 = x0 - P0 / dpdx;

			/* Weight computing */
			w1 = 2.0 / ((1.0 - x1 * x1) * dpdx * dpdx);

			/* Compute weight w0 on first iteration, needed for dw */
			if (j == 0) w0 = 2.0 / ((1.0 - x0 * x0) * dpdx * dpdx);

			dx = x0 - x1;
			dw = w0 - w1;

			x0 = x1;
			w0 = w1;
			j++;
		} while ((abs(dx) > eps || abs(dw) > eps) && j < 100);

		x[(m - 1) - (i - 1)] = x1;
		w[(m - 1) - (i - 1)] = w1;
	}

	return;
}

__global__ void gauss_legendre_method_Kernel(state A, state B, state* x, state* w, state*s) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	state Ax = A * x[idx];
	s[idx] = w[idx] * (fun3(B + Ax) + fun3(B - Ax));
}

__device__ void warpReduce_gausse_legendre(volatile state* sdata, int tid) {
	sdata[tid] += sdata[tid + 32];
	sdata[tid] += sdata[tid + 16];
	sdata[tid] += sdata[tid + 8];
	sdata[tid] += sdata[tid + 4];
	sdata[tid] += sdata[tid + 2];
	sdata[tid] += sdata[tid + 1];
}

__global__ void reduce_gausse_legendre(state* g_idata, state* g_odata) {
	extern __shared__ state sdata[TPB];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
	sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
	__syncthreads();

	for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
			__syncthreads();
		}
	}
	if (tid < 32) warpReduce_gausse_legendre(sdata, tid);
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}


state gauss_legendre_method_CUDA(state a, state b, int steps)
{
	state A, B, s;
	int i, m;

	m = (steps + 1) >> 1;

	size_t size = m * sizeof(state);
	state* x_d = NULL;
	state* w_d = NULL;
	state* s_d = NULL;

	state* s_h = (state*)malloc(size);
	state* x_h = (state*)malloc(size);
	state* w_h = (state*)malloc(size);

	cudaMalloc((void**)&x_d, size);
	cudaMalloc((void**)&w_d, size);
	cudaMalloc((void**)&s_d, size);

	gauss_legendre_tbl(steps, x_h, w_h, 1e-10);

	A = 0.5 * (b - a);
	B = 0.5 * (b + a);

	cudaMemcpy(w_d, w_h, size, cudaMemcpyHostToDevice);
	cudaMemcpy(x_d, x_h, size, cudaMemcpyHostToDevice);

	if (steps & 1) /* n - odd */
	{
		s = w_h[0] * (fun3(B));

		gauss_legendre_method_Kernel <<< (steps + 1023) / 1024, 1024 >>> (A, B, x_d, w_d, s_d);
		cudaMemcpy(s_h, s_d, size, cudaMemcpyDeviceToHost);
		for (i = 1; i < m; i++)
		{
			s += s_h[i];
		}

	}
	else { /* n - even */

		s = 0;
		gauss_legendre_method_Kernel << < (steps + 1023) / 1024, 1024 >> > (A, B, x_d, w_d, s_d);
		cudaMemcpy(s_h, s_d, size, cudaMemcpyDeviceToHost);
		for (i = 0; i < m; i++)
		{
			s += s_h[i];
		}
	}

	free(x_h);
	free(w_h);
	free(s_h);
	cudaFree(s_d); cudaFree(x_d); cudaFree(w_d);

	return A * s;
}

state gauss_legendre_method_CUDA_red(state a, state b, int steps)
{
	state A, B, s;
	int m;

	m = (steps + 1) >> 1;
	size_t size = m * sizeof(state);
	state* x_d = NULL;
	state* w_d = NULL;
	state* s_d = NULL;

	state* s_h = (state*)malloc(size);
	state* x_h = (state*)malloc(size);
	state* w_h = (state*)malloc(size);

	cudaMalloc((void**)&x_d, size);
	cudaMalloc((void**)&w_d, size);
	cudaMalloc((void**)&s_d, size);

	gauss_legendre_tbl(steps, x_h, w_h, 1e-10);

	A = 0.5 * (b - a);
	B = 0.5 * (b + a);

	cudaMemcpy(w_d, w_h, size, cudaMemcpyHostToDevice);
	cudaMemcpy(x_d, x_h, size, cudaMemcpyHostToDevice);

	if (steps & 1) /* n - odd */
	{
		s = w_h[0] * (fun3(B));

		gauss_legendre_method_Kernel <<< (steps + 1023) / 1024, 1024 >>> (A, B, x_d, w_d, s_d);

		cudaMemset(s_d, 0, sizeof(state));
		reduce_gausse_legendre <<< (int)(steps / pow(TPB, 1))+1, TPB / 2 >>> (s_d, s_d);
		cudaMemcpy(s_h, s_d, sizeof(state), cudaMemcpyDeviceToHost);

		s += s_h[0];
	}
	else { /* n - even */

		gauss_legendre_method_Kernel << < (steps + 1023) / 1024, 1024 >> > (A, B, x_d, w_d, s_d);
		reduce_gausse_legendre <<< (int)(steps / pow(TPB, 1)) + 1, TPB / 2 >>> (s_d, s_d);
		cudaMemcpy(s_h, s_d, sizeof(state), cudaMemcpyDeviceToHost);
		s = s_h[0];
	}

	free(x_h);
	free(w_h);
	free(s_h);
	cudaFree(s_d); cudaFree(x_d); cudaFree(w_d);

	return A * s;
}