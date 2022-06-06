#define _USE_MATH_DEFINES

#include <iostream>
#include <math.h>
#include <cmath>
#include <Bits.h>
#include <numeric>

#include <cuda_runtime.h>
#include <nvfunctional>

#include <chrono>
#include <map>

#define TPB 256


using namespace std;

/* Pozwala na szybk� zmian� pomi�dzy double oraz float
* double - wi�ksza dok�adno��, wolniejsze dzia�anie
* float  - mniejsza dok�adno��, szybsze dzia�anie
*/
typedef double state;

__device__ __host__ state funkcja(state x);