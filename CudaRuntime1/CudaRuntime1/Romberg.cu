#define _USE_MATH_DEFINES
#include <cmath>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <iomanip>

#include <cuda_runtime.h>
#include <nvfunctional>

#include "Romberg.cuh"
#include "Parameters.cu"

using namespace std;

extern __device__ __host__  state fun2(state x) {
	//return sin(x);
	return pow(exp(1.0), x);

	//przygotowana tablica, która zwraca tab[x]
}

state romberg_method(state a, state b, int N) {
	state area = 0;

	state* h = new state[N + 1];
	state** r = new state * [N + 1];

	for (int i = 0; i < N + 1; i++) {
		r[i] = new state[N + 1];
		h[i] = 0;
		for (int k = 0; k < N + 1; k++) {
			r[i][k] = 0;
		}
	}

	for (int i = 1; i < N + 1; ++i) {
		h[i] = (b - a) / pow(2, i - 1);
	}
	
	r[1][1] = h[1] / 2 * (fun2(a) + fun2(b));

	

	for (int i = 2; i < N + 1; ++i) {
		state coeff = 0;
		for (int k = 1; k <= pow(2, i - 2); ++k) {
			coeff += fun2(a + (2 * k - 1) * h[i]);
		}
		r[i][1] = 0.5 * (r[i - 1][1] + h[i - 1] * coeff);
	}

	for (int i = 2; i < N + 1; ++i) {
		for (int j = 2; j <= i; ++j) {
			r[i][j] = r[i][j - 1] + (r[i][j - 1] - r[i - 1][j - 1]) / (pow(4, j - 1) - 1);
		}
	}

	area = r[N][N];

	for (int i = 0; i < N + 1; i++) {
		delete[] r[i];
	}

	delete[] r;
	delete[] h;

	return area;
}


__global__ void romberg_method_Kernel_1(state a, state b, int max_eval, state* result)
{
	extern __shared__ state local_array[];
	
	state diff = (b - a) / gridDim.x;
	state step;

	b = a + (blockIdx.x + 1) * diff;
	a += blockIdx.x * diff;

	step = (b - a) / max_eval;
	
	for (int k = threadIdx.x; k < max_eval + 1; k += blockDim.x)
		local_array[k] = fun2(a + step * k);

	if (threadIdx.x < 13)
	{
		int inc = 1 << (12 - threadIdx.x);
		state sum = 0;
		for (int k = 0; k <= max_eval; k = k + inc)
		{
			sum += 2.0 * local_array[k];
		}
		sum -= (local_array[0] + local_array[max_eval]);
		sum *= (b - a) / (1 << (threadIdx.x + 1));
		local_array[threadIdx.x] = sum;
	}

	if (!threadIdx.x)
	{
		state romberg_table[13];
		for (int k = 0; k < 13; k++)
			romberg_table[k] = local_array[k];

		for (int col = 0; col < 12; col++)
		{
			for (int row = 12; row > col; row--)
			{
				romberg_table[row] = romberg_table[row] + (romberg_table[row] - romberg_table[row - 1]) / ((1 << (2 * col + 1)) - 1);
			}
		}
		result[blockIdx.x] = romberg_table[12];
	}

}

state romberg_method_CUDA_1(state a, state b, int N) {

		int numBlocks = N, numThreadsPerBlock = 1024, max_eval = 4096;

		size_t size = numBlocks * sizeof(state);

		state area = 0;

		state* result_d;
		state* result_h = (state*)malloc(size);

		cudaMalloc((void**)&result_d, size);

		cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
		romberg_method_Kernel_1 <<< numBlocks, numThreadsPerBlock, (max_eval + 1) * sizeof(state) >>> (a, b, max_eval, result_d);
		cudaDeviceSynchronize();

		cudaMemcpy(result_h, result_d, size, cudaMemcpyDeviceToHost);

		for (int k = 0; k < numBlocks; k++){
			area += result_h[k];}

		free(result_h);
		cudaFree(result_d);

		return area;
	}

__global__ void romberg_method_Kernel_2_1(state a, state b, state* h) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	h[idx] = (b - a) / pow(2, idx - 1);
}

__global__ void romberg_method_Kernel_2_2(state a, int i, state* h, state* coeff) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	coeff[idx] = fun2(a + (2 * idx - 1) * h[i]);
}

__global__ void romberg_method_Kernel_2_3(double* r, int Width) {

	int idx = blockIdx.x;
	int idy = threadIdx.x;

	if (idy * Width + idx < Width * Width && idx>2) {
			for (int j = 2; j <= Width; ++j) {
				if (idy >= j) {
					r[idy * Width + j] = threadIdx.x;//r[idy * Width + j - 1] + (r[idy * Width + j - 1] - r[(idy - 1) * Width + j - 1]) / (pow(4, j - 1) - 1);
				}
			}
			__syncthreads();
	}
}

state romberg_method_CUDA_2(state a, state b, int N) {

		state area = 0;

		int Width = N + 1;
		size_t size = Width * sizeof(state);
		size_t size_coeff = (int)(pow(2, N - 2)+2) * sizeof(state);

		state* h_h = NULL;
		state* coeff_h = NULL;
		state** r_h = NULL;

		h_h = (state*)malloc(size);
		coeff_h = (state*)malloc(size_coeff);
		r_h = new state * [Width];

		r_h[0] = new state[Width * Width];

		state* r_d = NULL;
		state* h_d = NULL;
		state* coeff_d = NULL;

		for (int i = 0; i < Width; i++) {
			r_h[i] = r_h[0] + i * Width; 
		}

		cudaMalloc(&h_d, size);
		cudaMemcpy(h_d, h_h, size, cudaMemcpyHostToDevice);
		romberg_method_Kernel_2_1 <<< (Width + 1023) / 1024, 1024 >>> (a, b, h_d);
		cudaMemcpy(h_h, h_d, size, cudaMemcpyDeviceToHost);

		r_h[1][1] = h_h[1] / 2 * (fun2(a) + fun2(b));

		cudaMalloc(&coeff_d, size_coeff);

		for (int i = 0; i <= pow(2, N - 2) + 1; i++) {
			coeff_h[i] = 0;
		}

		for (int i = 2; i < Width; ++i) {
			state coeff_sum = 0;

			cudaMemcpy(coeff_d, coeff_h, size_coeff, cudaMemcpyHostToDevice);
			romberg_method_Kernel_2_2 <<< (Width + 1023) / 1024, 1024 >>> (a, i, h_d, coeff_d);
			cudaMemcpy(coeff_h, coeff_d, size_coeff, cudaMemcpyDeviceToHost);

			for (int k = 1; k <= pow(2, i - 2); ++k) {
				coeff_sum += coeff_h[k];
				coeff_h[k] = 0;
			}
			r_h[i][1] = 0.5 * (r_h[i - 1][1] + h_h[i - 1] * coeff_sum);
		}

		//cudaMalloc(&r_d, size * size);
		//cudaMemcpy(r_d, r_h[0], size * size, cudaMemcpyHostToDevice);
		//romberg_method_Kernel_2_3 <<< Width, Width >>> (r_d, Width);
		//cudaMemcpy(r_h[0], r_d, size * size, cudaMemcpyDeviceToHost);

		for (int i = 2; i < Width; ++i) {
			for (int j = 2; j <= i; ++j) {
				r_h[i][j] = r_h[i][j - 1] + (r_h[i][j - 1] - r_h[i - 1][j - 1]) / (pow(4, j - 1) - 1);
			}
		}

		area = r_h[N][N];

		delete r_h[0];
		delete r_h;
		delete h_h;
		delete coeff_h;

		cudaFree(coeff_d);
		cudaFree(h_d);
		cudaFree(r_d);

		return area;
	
}
