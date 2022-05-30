#define _USE_MATH_DEFINES
#include <cmath>
#include <stdlib.h>
#include <float.h>
#include <iostream>
#include <Bits.h>
#include <numeric>

#include <cuda_runtime.h>
#include <nvfunctional>

#include "Simpson38.cuh"
#include "Parameters.cu"

__device__ __host__  state fun(state x) {
	//return sin(x);
	return pow(exp(1.0), x);

	//przygotowana tablica, która zwraca tab[x]
}

state simpson38_method(state a, state b, int steps) {

	state k = 0;
	state stepSize = (b - a) / steps;
	state integration = fun(a) + fun(b);

	for (int i = 1; i < steps; i++)
	{
		k = a + i * stepSize;
		if (i % 3 == 0) integration += 2 * (fun(k));
		else integration += 3 * (fun(k));
	}
	return integration * stepSize * 3.0 / 8.0;
}

__global__ void simpson38_method_Kernel(state a, state step, state* results)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	state x = a + idx * step;

	if (idx % 3 == 0) results[idx] = 2 * fun(x);
	else results[idx] = 3 * fun(x);
}

state simpson38_method_CUDA(state a, state b, int steps) {

	size_t size = steps * sizeof(state);

	state step = (b - a) / steps;
	state sum = fun(a) + fun(b);

	state* results_h = (state*)malloc(size);
	state* results_d = NULL;

	cudaMalloc((void**)&results_d, size);

	simpson38_method_Kernel <<< (steps + 1023) / 1024, 1024 >>> (a, step, results_d);

	cudaMemcpy(results_h, results_d, size, cudaMemcpyDeviceToHost);

	for (int i = 1; i < steps; i++) sum += results_h[i];

	free(results_h);
	cudaFree(results_d);

	return sum * step * 3 / 8;
}