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

#include "Simpson38.cuh"
#include "Parameters.cu"

__device__ __host__  state fun(state x) {
	//return sin(x);
	return pow(exp(1.0), x);
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

__device__ void warpReduce_simpson38(volatile state* sdata, int tid) {
	sdata[tid] += sdata[tid + 32];
	sdata[tid] += sdata[tid + 16];
	sdata[tid] += sdata[tid + 8];
	sdata[tid] += sdata[tid + 4];
	sdata[tid] += sdata[tid + 2];
	sdata[tid] += sdata[tid + 1];
}

__global__ void reduce_simpson38(state* g_idata, state* g_odata) {
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
	if (tid < 32) warpReduce_simpson38(sdata, tid);
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
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

state simpson38_method_CUDA_red(state a, state b, int steps) {

	size_t size = steps * sizeof(state);

	int total_steps = (int)pow(TPB, ceil(log(steps) / log(TPB)));
	size_t size_total = total_steps * sizeof(state);

	state step = (b - a) / steps;
	state sum = fun(a) + fun(b);

	state* results_h = (state*)malloc(size_total);
	state* results_d = NULL;
	state* results_d2 = NULL;

	cudaMalloc((void**)&results_d, size);
	cudaMalloc((void**)&results_d2, size_total);

	simpson38_method_Kernel << < (steps + 1023) / 1024, 1024 >> > (a, step, results_d);

	cudaMemcpy(results_h, results_d, size, cudaMemcpyDeviceToHost);
	memset(results_h + steps, 0, size_total - size);
	results_h[0] = 0;
	cudaMemcpy(results_d2, results_h, size_total, cudaMemcpyHostToDevice);

	int i = 1;
	while (total_steps / pow(TPB, i) >= 1) {
		reduce_simpson38 <<< (int)(total_steps / pow(TPB, i)), TPB / 2 >>> (results_d2, results_d2);
		i++;
	}

	cudaMemcpy(results_h, results_d2, sizeof(state), cudaMemcpyDeviceToHost);
	sum += results_h[0];

	free(results_h);
	cudaFree(results_d); cudaFree(results_d2);

	return sum * step * 3 / 8;
}