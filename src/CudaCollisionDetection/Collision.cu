#include"Collision.cuh"
#include "Ball.hpp"


//通用函数

__device__ float Dist(float x, float y, float z)
{
	return sqrt(x * x + y * y + z * z);
}

__device__ float Dist(Point& p)
{
	return sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
}

__device__ float Multiply(Point& a, Point& b)
{
	return (a.x * b.x + a.y * b.y + a.z * b.z);
}

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

/*
	描述：判断两个球是否相撞
	参数：球a，球b
	返回：是1，否0
*/
__device__ bool JudgeCollision(Ball& a, Ball& b)
{
	float dist = 0;
	float dist_x = a.CurrentPlace.x - b.CurrentPlace.x;
	float dist_y = a.CurrentPlace.y - b.CurrentPlace.y;
	float dist_z = a.CurrentPlace.z - b.CurrentPlace.z;
	dist = Dist(dist_x, dist_y, dist_z);
	if (dist < a.Radius + b.Radius)
	{
		return 1;
	}
	else
	{
		return 0;
	}
}

/*
	描述：两球相撞后更新速度
	参数：球a，球b
	返回：无
*/
__device__ void ChangeSpeed(Ball& a, Ball& b)
{
	//径向速度按照质量做变换，法向速度不变
	float dist = 0;
	float diff_x = b.CurrentPlace.x - a.CurrentPlace.x;
	float diff_y = b.CurrentPlace.y - a.CurrentPlace.y;
	float diff_z = b.CurrentPlace.z - a.CurrentPlace.z;
	dist = Dist(diff_x, diff_y, diff_z);

	//求径向，法向速度
	float rate_collide_a = (a.CurrentSpeed.x * diff_x + a.CurrentSpeed.y * diff_y + a.CurrentSpeed.z * diff_z) / dist / dist;
	float speed_collide_a_x = diff_x * rate_collide_a;
	float speed_collide_a_y = diff_y * rate_collide_a;
	float speed_collide_a_z = diff_z * rate_collide_a;

	float rate_collide_b = (b.CurrentSpeed.x * diff_x + b.CurrentSpeed.y * diff_y + b.CurrentSpeed.z * diff_z) / dist / dist;
	float speed_collide_b_x = diff_x * rate_collide_b;
	float speed_collide_b_y = diff_y * rate_collide_b;
	float speed_collide_b_z = diff_z * rate_collide_b;

	float unchanged_a_x = a.CurrentSpeed.x - speed_collide_a_x;
	float unchanged_a_y = a.CurrentSpeed.y - speed_collide_a_y;
	float unchanged_a_z = a.CurrentSpeed.z - speed_collide_a_z;

	float unchanged_b_x = b.CurrentSpeed.x - speed_collide_b_x;
	float unchanged_b_y = b.CurrentSpeed.y - speed_collide_b_y;
	float unchanged_b_z = b.CurrentSpeed.z - speed_collide_b_z;


	//假设b不动，a撞b，更新两者径向速度
	float speed_collide_new_a_x = (speed_collide_a_x * (a.Weight - b.Weight) + speed_collide_b_x * (2 * b.Weight)) / (a.Weight + b.Weight);
	float speed_collide_new_a_y = (speed_collide_a_y * (a.Weight - b.Weight) + speed_collide_b_y * (2 * b.Weight)) / (a.Weight + b.Weight);
	float speed_collide_new_a_z = (speed_collide_a_z * (a.Weight - b.Weight) + speed_collide_b_z * (2 * b.Weight)) / (a.Weight + b.Weight);

	float speed_collide_new_b_x = (speed_collide_a_x * (2 * a.Weight) + speed_collide_b_x * (b.Weight - a.Weight)) / (a.Weight + b.Weight);
	float speed_collide_new_b_y = (speed_collide_a_y * (2 * a.Weight) + speed_collide_b_y * (b.Weight - a.Weight)) / (a.Weight + b.Weight);
	float speed_collide_new_b_z = (speed_collide_a_z * (2 * a.Weight) + speed_collide_b_z * (b.Weight - a.Weight)) / (a.Weight + b.Weight);

	a.CurrentSpeed.x = speed_collide_new_a_x + unchanged_a_x;
	a.CurrentSpeed.y = speed_collide_new_a_y + unchanged_a_y;
	a.CurrentSpeed.z = speed_collide_new_a_z + unchanged_a_z;

	b.CurrentSpeed.x = speed_collide_new_b_x + unchanged_b_x;
	b.CurrentSpeed.y = speed_collide_new_b_y + unchanged_b_y;
	b.CurrentSpeed.z = speed_collide_new_b_z + unchanged_b_z;
}

/*
描述：在球之间的碰撞检测完成后，处理球的运动以及和边界的碰撞（并行）
参数：球列表，一次的时间，X范围(-X,X),Z范围(-Z,Z),Y范围(0,Y)，球个数
返回：无，但是更新球列表
*/
__global__ void UpdateBallsMove(Ball* balls, float TimeOnce, float XRange, float ZRange, float Height, int N)
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


//暴力算法相关函数
/*
描述：暴力算法处理碰撞检测和速度更新
参数：球列表，N个球
返回：无，但是更新球列表
*/
__global__ void HandleCollisionNaive(Ball* balls, int N)
{
	// 获取全局索引
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	// 步长
	int stride = blockDim.x * gridDim.x;
	for (int k = index; k < N * N; k += stride)
	{
		int i = k / N;
		int j = k % N;
		if(i < j)
		{
			if (JudgeCollision(balls[i], balls[j]))
			{
				ChangeSpeed(balls[i], balls[j]);
			}
		}
	}
}



/*
描述：GPU碰撞检测+运动更新主函数（暴力算法）
参数：球列表，一次的时间，X范围(-X,X),Z范围(-Z,Z),Y范围(0,Y)，球个数
返回：无，但是更新球列表
*/
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
	HandleCollisionNaive << < gridSize, blockSize >> > (balls_gpu, N);
	// 同步device 保证结果能正确访问
	cudaDeviceSynchronize();

	// 执行kernel
	UpdateBallsMove <<< gridSize, blockSize >>> (balls_gpu, TimeOnce, XRange, ZRange, Height, N);
	// 同步device 保证结果能正确访问
	cudaDeviceSynchronize();

	// 记录结果
	cudaMemcpy((void*)balls, (void*)balls_gpu, nBytes, cudaMemcpyDeviceToHost);

	// 释放内存
	cudaFree(balls_gpu);
}


//空间划分用的工具函数
/*
描述：计算前i和的算法
参数：原始数组，个数n
返回：原始数组变成前i个和数组
*/
__device__ void PrefixSum(uint32_t *values, unsigned int n) {
	int offset = 1;
	int a;
	uint32_t temp;

	// upsweep
	for (int d = n / 2; d; d /= 2) {
		__syncthreads();

		if (threadIdx.x < d) {
			a = (threadIdx.x * 2 + 1) * offset - 1;
			values[a + offset] += values[a];
		}

		offset *= 2;
	}

	if (!threadIdx.x) {
		values[n - 1] = 0;
	}

	// downsweep
	for (int d = 1; d < n; d *= 2) {
		__syncthreads();
		offset /= 2;

		if (threadIdx.x < d) {
			a = (threadIdx.x * 2 + 1) * offset - 1;
			temp = values[a];
			values[a] = values[a + offset];
			values[a + offset] += temp;
		}
	}
}

/*
描述：求和算法
参数：原始数组，输出
返回：输出变为和
*/
__device__ void dSumReduce(unsigned int *values, unsigned int *out) {
	// wait for the whole array to be populated
	__syncthreads();

	// sum by reduction, using half the threads in each subsequent iteration
	unsigned int threads = blockDim.x;
	unsigned int half = threads / 2;

	while (half) {
		if (threadIdx.x < half) {
			// only keep going if the thread is in the first half threads
			for (int k = threadIdx.x + half; k < threads; k += half) {
				values[threadIdx.x] += values[k];
			}

			threads = half;
		}

		half /= 2;

		// make sure all the threads are on the same iteration
		__syncthreads();
	}

	// only let one thread update the current sum
	if (!threadIdx.x) {
		atomicAdd(out, values[0]);
	}
}


//空间划分算法相关函数
/*
描述：初始化cells，objects数组，前者记录物体所在的格子信息（格子x，y，z的id，home还是phantom），后者记录物体id和home/phantom
参数：空的cell，phantom；球列表和个数，还有各种格子信息
返回：更新cells，objects数组和cell_num
*/
__global__ void InitCellKernel(uint32_t *cells, uint32_t *objects, Ball* balls, int N,
	float XRange, float ZRange, float Height, float GridSize, int GridX, int GridY, int GridZ, unsigned int* cell_num) 
{
	extern __shared__ unsigned int t[];
	unsigned int count = 0;

	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x)
	{
		int current_cell_id = i * 8; //每个球最多在8个格子内
		int cell_info = 0;
		int object_info = 0;
		int current_count = 0;
		float x = balls[i].CurrentPlace.x;
		float y = balls[i].CurrentPlace.y;
		float z = balls[i].CurrentPlace.z;
		float radius = balls[i].Radius;

		//找到home cell
		int hash_x = (x + XRange) / GridSize;
		int hash_y = (y) / GridSize;
		int hash_z = (z + ZRange) / GridSize;
		cell_info = hash_x << 17 | hash_y << 9 | hash_z << 1 | HOME_CELL;
		object_info = i << 1 | HOME_OBJECT;
		cells[current_cell_id] = cell_info;
		objects[current_cell_id] = object_info;
		current_cell_id++;
		count++;
		current_count++;

		//找phantom
		for (int dx = -1; dx <= 1; dx++)
		{
			for (int dy = -1; dy <= 1; dy++)
			{
				for (int dz = -1; dz <= 1; dz++)
				{
					int new_hash_x = hash_x + dx;
					int new_hash_y = hash_y + dy;
					int new_hash_z = hash_z + dz;

					//自己不考虑
					if (dx == 0 && dy == 0 && dz == 0)
					{
						continue;
					}

					//越界不考虑
					if (new_hash_x < 0 || new_hash_x >= GridX ||
						new_hash_y < 0 || new_hash_y >= GridY ||
						new_hash_z < 0 || new_hash_z >= GridZ)
					{
						continue;
					}

					float relative_x = 0;
					float relative_y = 0;
					float relative_z = 0;
					if (dx == 0)
					{
						relative_x = x;
					}
					else if (dx == -1)
					{
						relative_x = hash_x * GridSize - XRange;
					}
					else
					{
						relative_x = (hash_x + 1) * GridSize - XRange;
					}

					if (dz == 0)
					{
						relative_z = z;
					}
					else if (dz == -1)
					{
						relative_z = hash_z * GridSize - ZRange;
					}
					else
					{
						relative_z = (hash_z + 1) * GridSize - ZRange;
					}

					if (dy == 0)
					{
						relative_y = y;
					}
					else if (dy == -1)
					{
						relative_y = hash_y * GridSize;
					}
					else
					{
						relative_y = (hash_y + 1) * GridSize;
					}

					relative_x -= x;
					relative_y -= y;
					relative_z -= z;

					float dist = Dist(relative_x, relative_y, relative_z);
					if (dist < radius)
					{
						int cell_info = new_hash_x << 17 | new_hash_y << 9 | new_hash_z << 1 | PHANTOM_CELL;
						int object_info = i << 1 | PHANTOM_OBJECT;
						cells[current_cell_id] = cell_info;
						objects[current_cell_id] = object_info;
						current_cell_id++;
						count++;
						current_count++;
					}
				}
			}
		}

		//补齐
		while (current_count < 8)
		{

			cells[current_cell_id] = UINT32_MAX;
			objects[current_cell_id] = i << 2;
			current_cell_id++;
			current_count++;
		}

	}

	//计算有物体的格子个数
	t[threadIdx.x] = count;
	dSumReduce(t, cell_num);
}

/*
描述：初始化cells， objects数组的主函数
参数：空的cell，phantom；球列表和个数，还有各种格子信息，线程信息
返回：更新cells，objects数组和cell_num
 */
 unsigned int InitCells(uint32_t *cells, uint32_t *objects, Ball* balls, int N,
	 float XRange, float ZRange, float Height, float GridSize, int GridX, int GridY, int GridZ, 
	 unsigned int* cell_count_temp, unsigned int num_blocks, unsigned int threads_per_block) {
	 unsigned int cell_count;
	 cudaMemset(&cell_count, 0, sizeof(unsigned int));
	 InitCellKernel << <num_blocks, threads_per_block,
		 threads_per_block * sizeof(unsigned int) >> > (
			 cells, objects, balls, N, XRange, ZRange, Height, GridSize, GridX, GridY, GridZ, cell_count_temp);
	 cudaMemcpy(&cell_count, cell_count_temp, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	 return cell_count;
 }


/*
描述：对cells求前缀和
参数：cells，待更新前缀和，N个cell，偏移量
返回：更新前缀和
*/
 __global__ void GetRadixSum(uint32_t *cells, uint32_t *radix_sums, int N, int shift)
 {
	 int index = threadIdx.x + blockIdx.x * blockDim.x;
	 int stride = blockDim.x * gridDim.x;
	 int num_indices = 1 << RADIX_LENGTH;


	 //初始化
	 for (int i = index; i < num_indices; i++)
	 {
		 radix_sums[i] = 0;
	 }
	 __syncthreads();


	 //求和
	 for (int i = index; i < N; i += stride)
	 {
		 //非常重要，不这样做无法有效求和
		 for (int j = 0; j < blockDim.x; j++)
		 {
			 if (threadIdx.x % blockDim.x == j)
			 {
				 int current_radix_num = (cells[i] >> shift) & (num_indices - 1);
				 radix_sums[current_radix_num] ++;
			 }
		 }

	 }
	 __syncthreads();
	 //求前缀和
	 PrefixSum(radix_sums, num_indices);
	 __syncthreads();
}

 /*
 描述：重新分配元素
 参数：cells，object数组，他们待更新的分配结果temp，前缀和数组，N个元素，偏移量，每个线程处理几个cell
 返回：更新前缀和
 */
 __global__ void RearrangeCell(uint32_t *cells, uint32_t *objects, uint32_t *cells_temp, uint32_t *objects_temp, 
	 uint32_t *radix_sums, int N, int shift)
 {
	 int index = threadIdx.x + blockIdx.x * blockDim.x;
	 int stride = blockDim.x * gridDim.x;
	 int num_indices = 1 << RADIX_LENGTH;

	 if (index != 0) return;
	 //分配
	 for (int i = 0; i < N; i ++ )
	 {
		int current_radix_num = (cells[i] >> shift) & (num_indices - 1);
		cells_temp[radix_sums[current_radix_num]] = cells[i];
		objects_temp[radix_sums[current_radix_num]] = objects[i];
		radix_sums[current_radix_num] ++;
	 }
 }


/*
描述：对cell，object做基数排序
参数：cell，object数组；他们的temp形式用于排序；待求的前缀和数组；cell个数；线程情况
返回：无，但是更新cell，object数组
*/
void SortCells(uint32_t *cells, uint32_t *objects, uint32_t *cells_temp, uint32_t *objects_temp,
	uint32_t *radix_sums, int N, unsigned int num_blocks, unsigned int threads_per_block)
{
	uint32_t *cells_swap;
	uint32_t *objects_swap;
	for (int i = 0; i < 32; i += RADIX_LENGTH)
	{
		//求前缀和
		GetRadixSum <<< num_blocks, threads_per_block >>> (cells, radix_sums, N, i);

		//用前缀和重新分配
		RearrangeCell << < num_blocks, threads_per_block >> > (cells, objects, cells_temp, objects_temp,
			radix_sums, N, i);
		
		//交换原始和temp
		cells_swap = cells;
		cells = cells_temp;
		cells_temp = cells_swap;
		objects_swap = objects;
		objects = objects_temp;
		objects_temp = objects_swap;
		

		
	}
}



/*
描述：空间划分算法处理碰撞检测和速度更新（主函数）
参数：球列表，X范围(-X,X),Z范围(-Z,Z),Y范围(0,Y)，格子大小，X格子个数，Y格子个数，Z格子个数，N个球
返回：无，但是更新球列表
*/
void HandleCollisionGrid(Ball* balls, float XRange, float ZRange, float Height, 
	float GridSize, int GridX, int GridY, int GridZ, int N,
	unsigned int num_blocks, unsigned int threads_per_block)
{

	//申请内存
	unsigned int cell_size = N * 8 * sizeof(uint32_t);
	unsigned int num_cells_occupied;
	/*unsigned int num_collisions;
	unsigned int num_collisions_cpu;
	unsigned int num_tests;
	unsigned int num_tests_cpu;*/

	uint32_t *cells_gpu;
	uint32_t *cells_gpu_temp;
	uint32_t *objects_gpu;
	uint32_t *objects_gpu_temp;
	unsigned int *temp_gpu;
	uint32_t *radix_sums_gpu;

	int num_radices = 1 << RADIX_LENGTH;

	cudaMalloc((void **)&cells_gpu, cell_size);
	cudaMalloc((void **)&cells_gpu_temp, cell_size);
	cudaMalloc((void **)&objects_gpu, cell_size);
	cudaMalloc((void **)&objects_gpu_temp, cell_size);
	cudaMalloc((void **)&temp_gpu, 2 * sizeof(unsigned int));
	cudaMalloc((void **)&radix_sums_gpu, num_radices * sizeof(uint32_t));


	
	//初始化cell和object
	num_cells_occupied = InitCells(cells_gpu, objects_gpu, balls, N,
		XRange, ZRange, Height, GridSize, GridX, GridY, GridZ,
		temp_gpu, num_blocks, threads_per_block);

	//基数排序
	SortCells(cells_gpu, objects_gpu, cells_gpu_temp, objects_gpu_temp, radix_sums_gpu, 
		8 * N, num_blocks, threads_per_block);
	

	/*num_collisions = cudaCellCollide(d_cells, d_objects, d_positions,
		d_velocities, d_dims, num_objects,
		num_cells, d_temp, &num_tests, num_blocks,
		threads_per_block);*/
	

	

	cudaFree(temp_gpu);
	cudaFree(cells_gpu);
	cudaFree(cells_gpu_temp);
	cudaFree(objects_gpu);
	cudaFree(objects_gpu_temp);
	cudaFree(radix_sums_gpu);
}


/*
描述：GPU碰撞检测+运动更新主函数（空间划分算法）
参数：球列表，一次的时间，X范围(-X,X),Z范围(-Z,Z),Y范围(0,Y)，一个格子大小，X,Y,Z的格子个数，球个数
返回：无，但是更新球列表
*/
void UpdateBallsGridGPU(Ball* balls, float TimeOnce, float XRange, float ZRange, float Height, 
	float GridSize, int GridX, int GridY, int GridZ, int N)
{
	//设置，计算需要多少block和thread
	unsigned int num_blocks = 1;
	unsigned int threads_per_block = 512;
	unsigned int object_size = (N - 1) / threads_per_block + 1;
	if (object_size < num_blocks) {
		num_blocks = object_size;
	}

	Ball* balls_gpu;
	unsigned int nBytes = N * sizeof(Ball);
	cudaMalloc((void**)&balls_gpu, nBytes);


	// 初始化数据
	cudaMemcpy((void*)balls_gpu, (void*)balls, nBytes, cudaMemcpyHostToDevice);

	// 执行kernel
	HandleCollisionGrid(balls_gpu, XRange, ZRange, Height, GridSize, GridX, GridY, GridZ, N, num_blocks, threads_per_block);
	// 同步device 保证结果能正确访问
	cudaDeviceSynchronize();

	// 执行kernel
	UpdateBallsMove << < num_blocks, threads_per_block>> > (balls_gpu, TimeOnce, XRange, ZRange, Height, N);
	// 同步device 保证结果能正确访问
	cudaDeviceSynchronize();

	// 记录结果
	cudaMemcpy((void*)balls, (void*)balls_gpu, nBytes, cudaMemcpyDeviceToHost);

	// 释放内存
	cudaFree(balls_gpu);
}