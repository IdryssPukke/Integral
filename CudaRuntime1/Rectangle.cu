#define _USE_MATH_DEFINES

#include <iostream>
#include <math.h>
#include <cmath>
#include <Bits.h>
#include <numeric>
#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>

#include <cuda_runtime.h>
#include <nvfunctional>
#include "device_launch_parameters.h"

#include <chrono>
#include <tuple>

#include "Rectangle.cuh"
#include "Parameters.cu"



__device__ __host__ state funkcja(state x) {
	//return sin(x);
	return pow(exp(1.0), x);
}

state rectangle_method(state a, state b, int n) {
	state step = (b - a) / n;
	state area = 0;

	for (int i = 0; i < n; ++i) {
		area += step * funkcja(a + i * step + 0.5 * step);
	}
	return area;
}

__global__ void rectangle_method_Kernel(state a, state* results, state step)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	results[idx] = step * funkcja(a + idx * step + 0.5 * step);
}

__device__ void warpReduce_rectangle(volatile state* sdata, int tid) {
	sdata[tid] += sdata[tid + 32];
	sdata[tid] += sdata[tid + 16];
	sdata[tid] += sdata[tid + 8];
	sdata[tid] += sdata[tid + 4];
	sdata[tid] += sdata[tid + 2];
	sdata[tid] += sdata[tid + 1];
}

__global__ void reduce_rectangle(state* g_idata, state* g_odata) {
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
		if (tid < 32) warpReduce_rectangle(sdata, tid);
		if (tid == 0) g_odata[blockIdx.x] = sdata[0];

}

__global__ void reduce_rectangle_test(state* g_idata, state* g_odata) {
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
		if (tid < 32) warpReduce_rectangle(sdata, tid);
		if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

state rectangle_method_CUDA(state a, state b, int steps) {

	size_t size = steps * sizeof(state);
	state step = (b - a) / steps;
	state sum = 0;

	state* results_h = (state*)malloc(size);
	state* results_d = NULL;

	cudaMalloc(&results_d, size);

	rectangle_method_Kernel <<< (steps + 1023) / 1024, 1024 >>> (a, results_d, step);

	cudaMemcpy(results_h, results_d, size, cudaMemcpyDeviceToHost);

	for (int i = 0; i < steps; i++) {
		sum += results_h[i];
	}

	free(results_h);
	cudaFree(results_d);

	return sum;
}

state rectangle_method_CUDA_red(state a, state b, int steps) {

	size_t size = steps * sizeof(state);
	state step = (b - a) / steps;
	state sum = 0;

	state* results_d2 = NULL;
	state* results_d = NULL;

	int total_steps = steps;

	total_steps = (int)pow(TPB, ceil(log(steps) / log(TPB)));

	size_t size_total = total_steps * sizeof(state);
	state* results_h = (state*)malloc(size_total);

	cudaMalloc(&results_d, size_total);
	cudaMalloc(&results_d2, size);

	rectangle_method_Kernel << < (steps + 1023) / 1024, 1024 >> > (a, results_d2, step);

	cudaMemcpy(results_h, results_d2, size, cudaMemcpyDeviceToHost);
	memset(results_h + steps, 0, size_total - size);
	cudaMemcpy(results_d, results_h, size_total, cudaMemcpyHostToDevice);

	int i = 1;
	while (total_steps / pow(TPB, i) >= 1) {
		reduce_rectangle <<< (int)(total_steps / pow(TPB, i)), TPB / 2 >>> (results_d, results_d);
		i++;
	}

	cudaMemcpy(results_h, results_d, sizeof(state), cudaMemcpyDeviceToHost);
	sum = results_h[0];

	free(results_h);
	cudaFree(results_d), cudaFree(results_d2);
	return sum;
}

state rectangle_method_CUDA_red2(state a, state b, int steps) {

	size_t size = steps * sizeof(state);
	state step = (b - a) / steps;
	state sum = 0;

	state* results_h = (state*)malloc(size);
	state* results_d = NULL;
	state* results_d2 = NULL;


	cudaMalloc(&results_d, size);


	rectangle_method_Kernel <<< (steps + 1023) / 1024, 1024 >>> (a, results_d, step);
	cudaMemcpy(results_h, results_d, size, cudaMemcpyDeviceToHost);
	for (int i = 0; i < steps; i++) {
		cout<<results_h[i]<<" ";
	}

	reduce_rectangle_test <<< (int)(steps / pow(TPB, 1))+1, TPB / 2 >>> (results_d, results_d);

	cudaMemcpy(results_h, results_d, size, cudaMemcpyDeviceToHost);
	sum = results_h[0];
	cout << endl;
	for (int i = 0; i < steps; i++) {
		cout<<results_h[i]<<" ";
	}

	free(results_h);
	cudaFree(results_d); cudaFree(results_d2);
	cout<<sum;
	return sum;
}