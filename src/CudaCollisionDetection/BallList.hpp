#pragma once
#include<math.h>
#include<freeglut/glut.h>
#include<vector>
#include<time.h>
#include"Point.hpp"
#include"Board.hpp"
#include"Ball.hpp"
#include"Collision.cuh"
using namespace std;

#define NAIVE_CPU 0
#define NAIVE_GPU 1
#define FAST_CPU 2
#define FAST_GPU 3

class BallList
{
public:
	Ball* balls;
	BallList() {}
	float XRange;
	float ZRange;
	float Height;
	int Num;
	int NBalls;
	float MaxRadius;
	float TimeOnce;
	int Mode;

	/*
	描述：初始化位置信息
	参数：x范围（实际是-x到x），y范围（0到y），z范围（-z到z），每个轴上球个数（实际num的立方个球），球最大半径，模式
	*/
	void Init(float x, float y, float z, int num, float max_radius, float time_once, int mode)
	{
		XRange = x;
		ZRange = z;
		Height = y;
		Num = num;
		MaxRadius = max_radius;
		TimeOnce = time_once;
		Mode = mode;
		NBalls = num * num * num;
		balls = new Ball[NBalls];
	}

	void InitBalls()
	{
		//小球的纹理，材质，颜色
		GLfloat color[3] = { 1.0, 0.0, 0.0 };
		GLfloat ambient[3] = { 0.4, 0.2, 0.2 };
		GLfloat diffuse[3] = { 1, 0.8, 0.8 };
		GLfloat specular[3] = { 0.5, 0.3, 0.3 };
		GLfloat shininess = 10;
		int complexity = 40;

		float diff_x = (2 * XRange - 2 * MaxRadius) / (Num - 1);
		float diff_z = (2 * ZRange - 2 * MaxRadius) / (Num - 1);
		float diff_y = (Height - 2 * MaxRadius) / (Num - 1);

		for (int i = 0; i < Num; i++)
		{
			for (int j = 0; j < Num; j++)
			{
				for (int k = 0; k < Num; k++)
				{	
					
					float place_x = diff_x * i + MaxRadius - XRange;
					float place_z = diff_z * j + MaxRadius - ZRange;
					float place_y = diff_y * k + MaxRadius;
					
					int index = i * Num * Num + j * Num + k;
					balls[index].InitColor(color, ambient, diffuse, specular, shininess, complexity);
					float speed_x = ((rand() % 201) / 100.0f - 1.0f) * 10;
					float speed_y = ((rand() % 201) / 100.0f - 1.0f) * 10;
					float speed_z = ((rand() % 201) / 100.0f - 1.0f) * 10;
					float radius = ((rand() % 51) / 100.0f + 0.5f) * MaxRadius;
					balls[index].InitPlace(place_x, place_y, place_z, radius, speed_x, speed_y, speed_z);
				}
			}

		}
	}

	/*
		描述：绘制所有球
		参数：无
		返回：无
	*/
	void DrawBalls()
	{
		for (int i = 0; i < NBalls; i++)
		{
			balls[i].DrawSelf();
		}
	}

	/*
		描述：判断两个球是否相撞
		参数：球a，球b
		返回：是1，否0
	*/
	bool JudgeCollision(Ball& a, Ball& b)
	{
		float dist = (a.CurrentPlace - b.CurrentPlace).Dist();
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
	void ChangeSpeed(Ball& a, Ball& b)
	{
		//径向速度按照质量做变换，法向速度不变
		Point diff = b.CurrentPlace - a.CurrentPlace;
		float dist = diff.Dist();
		
		//求径向，法向速度
		Point speed_collide_a = diff * (a.CurrentSpeed * diff / dist / dist);
		Point speed_collide_b = diff * (b.CurrentSpeed * diff / dist / dist);
		Point unchanged_a = a.CurrentSpeed - speed_collide_a;
		Point unchanged_b = b.CurrentSpeed - speed_collide_b;
		
		//假设b不动，a撞b，更新两者径向速度
		Point speed_collide_new_a = (speed_collide_a * (a.Weight - b.Weight) + speed_collide_b * (2 * b.Weight)) / (a.Weight + b.Weight);
		Point speed_collide_new_b = (speed_collide_a * (2 * a.Weight) + speed_collide_b * (b.Weight - a.Weight)) / (a.Weight + b.Weight);
		Point speed_new_a = speed_collide_new_a + unchanged_a;
		Point speed_new_b = speed_collide_new_b + unchanged_b;
		a.CurrentSpeed = speed_new_a;
		b.CurrentSpeed = speed_new_b;
	}

	//更新球的运动--主函数
	void UpdateBalls()
	{
		static int total_num = 0;
		static float total_time = 0;

		clock_t start, end;
		start = clock();

		if (Mode == NAIVE_CPU)
		{
			CollisionNaive();
			UpdateBallsMove();
		}
		else if (Mode == NAIVE_GPU)
		{
			UpdateBallsNaiveGPU(balls, TimeOnce, XRange, ZRange, Height, NBalls);
		}
		else if (Mode == FAST_CPU)
		{
			CollisionOctTree();
			UpdateBallsMove();
		}
		else if (Mode == FAST_GPU)
		{
			//UpdateBallsOctTreeGPU(balls, TimeOnce, XRange, ZRange, Height, NBalls);
		}
		end = clock();
		float duration = float(end - start) / CLOCKS_PER_SEC * 1000;
		total_num++;
		total_time += duration;
		if (total_num == 10000)
		{
			float average_time = total_time / total_num;
			cout << total_num << "次碰撞检测平均耗时" << average_time << "ms" << endl;
			total_num = 0;
			total_time = 0;
		}
	}

	/*
		描述：在球之间的碰撞检测完成后，处理球的运动以及和边界的碰撞（串行）
		参数：无
		返回：无
	*/
	void UpdateBallsMove()
	{
		for (int i = 0; i < NBalls; i++)
		{
			balls[i].Move(TimeOnce, XRange, ZRange, Height);
		}
	}


	/*
	描述：球之间碰撞检测（n^2算法，串行）
	参数：无
	返回：无
	*/
	void CollisionNaive()
	{
		for (int i = 0; i < NBalls - 1; i++)
		{
			for (int j = i + 1; j < NBalls; j++)
			{
				if (JudgeCollision(balls[i], balls[j]))
				{
					ChangeSpeed(balls[i], balls[j]);
				}
			}
		}
	}

	/*
	描述：球之间碰撞检测（八叉树加速，串行）
	参数：无
	返回：无
	*/
	void CollisionOctTree()
	{

	}

};