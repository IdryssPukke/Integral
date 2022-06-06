#define _USE_MATH_DEFINES
#include <cmath>
#include <stdlib.h>
#include <float.h>
#include <iostream>
#include <Bits.h>
#include <numeric>

#include <cuda_runtime.h>
#include <nvfunctional>
#include "device_launch_parameters.h"

#include "Simpson.cuh"
#include "Parameters.cu"

__device__ __host__  state fun1(state x) {
	//return sin(x);
	return pow(exp(1.0), x);
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
	state x = a + (idx+1) * step;

	st[idx] = fun1(x - step / 2);
	if (idx < steps) s[idx] = fun1(x);
}

__device__ void warpReduce_simpson(volatile state* sdata, int tid) {
	sdata[tid] += sdata[tid + 32];
	sdata[tid] += sdata[tid + 16];
	sdata[tid] += sdata[tid + 8];
	sdata[tid] += sdata[tid + 4];
	sdata[tid] += sdata[tid + 2];
	sdata[tid] += sdata[tid + 1];
}

__global__ void reduce_simpson(state* g_idata, state* g_odata) {
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
	if (tid < 32) warpReduce_simpson(sdata, tid);
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

state simpson_method_CUDA(state a, state b, int steps) {
	size_t size = steps * sizeof(state);

	state step = (b - a) / (steps);

	state sum_st = 0;
	state sum_s = 0;

	state* s_h = (state*)malloc(size);
	state* st_h = (state*)malloc(size);

	state* s_d = NULL;
	state* st_d = NULL;

	cudaMalloc((void**)&s_d, size);
	cudaMalloc((void**)&st_d, size);

	simpson_method_Kernel <<< (steps + 1023) / 1024, 1024 >>> (a, steps, step, st_d, s_d);

	cudaMemcpy(s_h, s_d, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(st_h, st_d, size, cudaMemcpyDeviceToHost);

	for (int i = 0; i < steps; i++) {
		sum_s += s_h[i];
		sum_st += st_h[i];
	}
	sum_s -= s_h[steps-1];

	free(s_h); free(st_h);
	cudaFree(s_d); cudaFree(st_d);

	return step / 6 * (funkcja(a) + funkcja(b) + 2 * sum_s + 4 * sum_st);
}


state simpson_method_CUDA_red(state a, state b, int steps) {
	size_t size = steps * sizeof(state);

	int total_steps = steps;

	if (steps % TPB != 0) {
		total_steps = (int)pow(TPB, ceil(log(steps) / log(TPB)));
	}

	size_t size_total = total_steps * sizeof(state);

	state step = (b - a) / (steps);
	state sum_st = 0;
	state sum_s = 0;

	state* results_h = (state*)malloc(size_total);

	state* s_d = NULL;
	state* st_d = NULL;
	state* results_d = NULL;

	cudaMalloc((void**)&s_d, size);
	cudaMalloc((void**)&st_d, size);
	cudaMalloc((void**)&results_d, size_total);

	simpson_method_Kernel <<< (steps + 1023) / 1024, 1024 >>> (a, steps, step, st_d, s_d);

	cudaMemcpy(results_h, s_d, size, cudaMemcpyDeviceToHost);
	memset(results_h + steps, 0, size_total - size);
	results_h[steps - 1] = 0;
	cudaMemcpy(results_d, results_h, size_total, cudaMemcpyHostToDevice);

	int i = 1;
	while (total_steps / pow(TPB, i) >= 1) {
		reduce_simpson <<< (int)(total_steps / pow(TPB, i)), TPB / 2 >>> (results_d, results_d);
		i++;
	}

	cudaMemcpy(results_h, results_d, sizeof(state), cudaMemcpyDeviceToHost);
	sum_s = results_h[0];

	cudaMemcpy(results_h, st_d, size, cudaMemcpyDeviceToHost);
	memset(results_h + steps, 0, size_total - size);
	cudaMemcpy(results_d, results_h, size_total, cudaMemcpyHostToDevice);

	i = 1;
	while (total_steps / pow(TPB, i) >= 1) {
		reduce_simpson <<< (int)(total_steps / pow(TPB, i)), TPB / 2 >>> (results_d, results_d);
		i++;
	}

	cudaMemcpy(results_h, results_d, sizeof(state), cudaMemcpyDeviceToHost);
	sum_st = results_h[0];

	free(results_h);
	cudaFree(s_d); cudaFree(st_d); cudaFree(results_d);

	return step / 6 * (funkcja(a) + funkcja(b) + 2 * sum_s + 4 * sum_st);
}