#define _USE_MATH_DEFINES

#include <iostream>
#include <math.h>
#include <cmath>
#include <Bits.h>
#include <numeric>

#include <cuda_runtime.h>
#include <nvfunctional>

#include <chrono>
#include <tuple>

#include "Rectangle.cuh"
#include "Parameters.cu"

__device__ __host__ state func(state x) {
	//return sin(x);
	return pow(exp(1.0), x);

	//przygotowana tablica, która zwraca tab[x]
}

state rectangle_method(state a, state b, int n) {
	state step = (b - a) / n;
	state area = 0;

	for (int i = 0; i < n; ++i) {
		area += step * func(a + i * step + 0.5 * step);
	}
	return area;
}

__global__ void rectangle_method_Kernel(state a, state* results, state step)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	results[idx] = step * func(a + idx * step + 0.5 * step);

}

state rectangle_method_CUDA(state a, state b, int steps) {
	size_t size = steps * sizeof(state);
	state step = (b - a) / steps;
	state sum = 0;

	state* results_h = (state*)malloc(size);
	state* results_d = NULL;

	cudaMalloc((void**)&results_d, size);

	rectangle_method_Kernel <<< (steps + 1023) / 1024, 1024 >>> (a, results_d, step);

	cudaMemcpy(results_h, results_d, size, cudaMemcpyDeviceToHost);

	for (int i = 0; i < steps; i++) sum += results_h[i];

	free(results_h);
	cudaFree(results_d);

	return sum;
}