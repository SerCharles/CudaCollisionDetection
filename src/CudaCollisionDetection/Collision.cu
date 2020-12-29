#include"Collision.cuh"
#include "Ball.hpp"


/*
	描述：处理与边界相撞
	参数：X范围（-X, X), Z范围(-Z, Z), Y范围(0, Y)
	返回：无
*/
__device__ void HandleCollisionBoard(Ball& ball, float XRange, float ZRange, float Height)
{
	if (ball.CurrentPlace.x - ball.Radius < -XRange)
	{
		ball.CurrentPlace.x = -XRange + ball.Radius;
		ball.CurrentSpeed.x = -ball.CurrentSpeed.x;
	}
	else if (ball.CurrentPlace.x + ball.Radius > XRange)
	{
		ball.CurrentPlace.x = XRange - ball.Radius;
		ball.CurrentSpeed.x = -ball.CurrentSpeed.x;
	}
	if (ball.CurrentPlace.z - ball.Radius < -ZRange)
	{
		ball.CurrentPlace.z = -ZRange + ball.Radius;
		ball.CurrentSpeed.z = -ball.CurrentSpeed.z;
	}
	else if (ball.CurrentPlace.z + ball.Radius > ZRange)
	{
		ball.CurrentPlace.z = ZRange - ball.Radius;
		ball.CurrentSpeed.z = -ball.CurrentSpeed.z;
	}
	if (ball.CurrentPlace.y - ball.Radius < 0)
	{
		ball.CurrentPlace.y = ball.Radius;
		ball.CurrentSpeed.y = -ball.CurrentSpeed.y;
	}
	else if (ball.CurrentPlace.y + ball.Radius > Height)
	{
		ball.CurrentPlace.y = Height - ball.Radius;
		ball.CurrentSpeed.y = -ball.CurrentSpeed.y;
	}
}



/*
	描述：处理小球自行运动和与边界碰撞
	参数：单次运动时间，X范围（-X, X), Z范围(-Z, Z), Y范围(0, Y)
	返回：无
*/
__device__ void BallMove(Ball& ball, float time, float XRange, float ZRange, float Height)
{
	ball.CurrentPlace.x = ball.CurrentPlace.x + ball.CurrentSpeed.x * time;
	ball.CurrentPlace.y = ball.CurrentPlace.y + ball.CurrentSpeed.y * time;
	ball.CurrentPlace.z = ball.CurrentPlace.z + ball.CurrentSpeed.z * time;
	HandleCollisionBoard(ball, XRange, ZRange, Height);
}

__global__ void UpdateBallsMoveNaive(Ball* balls, float TimeOnce, float XRange, float ZRange, float Height, int N)
{
	// 获取全局索引
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	// 步长
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < N; i += stride)
	{
		BallMove(balls[i], TimeOnce, XRange, ZRange, Height);
	}

}

void UpdateBallsNaiveGPU(Ball* balls, float TimeOnce, float XRange, float ZRange, float Height, int N)
{
	// 申请托管内存
	int nBytes = N * sizeof(Ball);
	Ball* balls_gpu;
	cudaMallocManaged((void**)&balls_gpu, nBytes);

	// 初始化数据
	cudaMemcpy((void*)balls_gpu, (void*)balls, nBytes, cudaMemcpyHostToDevice);

	// 定义kernel的执行配置
	dim3 blockSize(256);
	dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
	// 执行kernel
	UpdateBallsMoveNaive <<< gridSize, blockSize >>> (balls_gpu, TimeOnce, XRange, ZRange, Height, N);

	// 同步device 保证结果能正确访问
	cudaDeviceSynchronize();

	// 记录结果
	cudaMemcpy((void*)balls, (void*)balls_gpu, nBytes, cudaMemcpyHostToDevice);

	// 释放内存
	cudaFree(balls);
}