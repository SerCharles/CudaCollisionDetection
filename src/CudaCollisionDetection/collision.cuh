#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Ball.hpp"


#define HOME_CELL 0x00
#define PHANTOM_CELL 0x01
#define HOME_OBJECT 0x01
#define PHANTOM_OBJECT 0x00

void UpdateBallsNaiveGPU(Ball* balls, float TimeOnce, float XRange, float ZRange, float Height, int N);

void UpdateBallsGridGPU(Ball* balls, float TimeOnce, float XRange, float ZRange, float Height, 
	float GridSize, int GridX, int GridY, int GridZ, int N,
	unsigned int num_blocks, unsigned int threads_per_block);