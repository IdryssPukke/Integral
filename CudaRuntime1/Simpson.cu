#define _USE_MATH_DEFINES
#include <cmath>
#include <stdlib.h>
#include <float.h>
#include <iostream>
#include <Bits.h>
#include <numeric>

#include <cuda_runtime.h>
#include <nvfunctional>

#include "Simpson.cuh"
#include "Parameters.cu"

__device__ __host__  state fun1(state x) {
	//return sin(x);
	return pow(exp(1.0), x);

	//przygotowana tablica, która zwraca tab[x]
}

state simpson_method(state a, state b, int n) {
	state s = 0, st = 0, x;
	state step = (b - a) / n;
	for (int i = 1; i <= n; i++)
	{
		x = a + i * step;
		st += fun1(x - step / 2);
		if (i < n) s += fun1(x);
	}

	return step / 6 * (fun1(a) + fun1(b) + 2 * s + 4 * st);
}

__global__ void simpson_method_Kernel(state a, int steps, state step, state* st, state* s)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	state x = a + idx * step;

	st[idx] = fun1(x - step / 2);
	if (idx < steps) s[idx] = fun1(x);
}

state simpson_method_CUDA(state a, state b, int steps) {
	size_t size = (steps + 1) * sizeof(state);

	state step = (b - a) / steps;

	state sum_st = 0;
	state sum_s = 0;

	state* s_h = (state*)malloc(size);
	state* st_h = (state*)malloc(size);

	state* s_d = NULL;
	state* st_d = NULL;

	cudaMalloc((void**)&s_d, size);
	cudaMalloc((void**)&st_d, size);

	simpson_method_Kernel <<< (steps + 1023) / 1024, 1024 >>> (a, steps, step, st_d, s_d);

	cudaMemcpy(s_h, s_d, sizeof(state) * (steps + 1), cudaMemcpyDeviceToHost);
	cudaMemcpy(st_h, st_d, sizeof(state) * (steps + 1), cudaMemcpyDeviceToHost);

	for (int i = 1; i <= steps; i++) {
		sum_s += s_h[i];
		sum_st += st_h[i];
	}

	free(s_h); free(st_h);
	cudaFree(s_d); cudaFree(st_d);

	return step / 6 * (fun1(a) + fun1(b) + 2 * sum_s + 4 * sum_st);
}