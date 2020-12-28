#pragma once
#include<math.h>
#include<freeglut/glut.h>
#include"Point.hpp"
#include"Board.hpp"
using namespace std;

#pragma once
class MovingBall
{
public:
	//位置，速度信息
	Point CurrentPlace;
	float Radius;
	Point CurrentSpeed;

	//材质，纹理，颜色信息
	GLfloat Color[3] = { 0, 0, 0 }; //颜色
	GLfloat Ambient[4] = { 0, 0, 0, 0 }; //环境光
	GLfloat Diffuse[4] = { 0, 0, 0, 0 }; //漫反射
	GLfloat Specular[4] = { 0, 0, 0, 0 }; //镜面反射
	GLfloat Shininess[4] = { 0 }; //镜面指数
public:
	MovingBall(){}

	//初始化位置，速度信息
	void InitPlace(float x, float z, float radius, float speed_x, float speed_z)
	{
		CurrentPlace.SetPlace(x, radius, z);
		CurrentSpeed.SetPlace(speed_x, 0, speed_z);
		Radius = radius;
	}

	//初始化颜色，纹理，材质信息
	void InitColor(GLfloat color[], GLfloat ambient[], GLfloat diffuse[], GLfloat specular[], GLfloat shininess)
	{
		for (int i = 0; i < 3; i++)
		{
			Color[i] = color[i];
			Ambient[i] = ambient[i];
			Diffuse[i] = diffuse[i];
			Specular[i] = specular[i];
		}
		//透明度：1
		Ambient[3] = 1.0;
		Diffuse[3] = 1.0;
		Specular[3] = 1.0;
		Shininess[0] = shininess;
	}

	//求点到球心距离
	float GetDistance(Point p)
	{ 
		return sqrt((p - CurrentPlace) * (p - CurrentPlace));
	}

	//处理移动
	void Move(float time)
	{
		float dx = CurrentSpeed.x * time;
		float dy = CurrentSpeed.y * time;
		float dz = CurrentSpeed.z * time;
		float new_x = dx + CurrentPlace.x;
		float new_y = dy + CurrentPlace.y;
		float new_z = dz + CurrentPlace.z;
		CurrentPlace.SetPlace(new_x, new_y, new_z);
	}

	/*
		描述：处理与边界相撞
		参数：X范围（-X,X),Z范围(-Z,Z)
		返回：无
	*/
	void HandleCollisionBoard(float XRange, float ZRange)
	{
		if (CurrentPlace.x - Radius < -XRange)
		{
			cout << "球和边界碰撞" << endl;
			CurrentPlace.x = -XRange + Radius;
			CurrentSpeed.x = -CurrentSpeed.x;
		}
		else if (CurrentPlace.x + Radius > XRange)
		{
			cout << "球和边界碰撞" << endl;
			CurrentPlace.x = XRange - Radius;
			CurrentSpeed.x = -CurrentSpeed.x;
		}
		if (CurrentPlace.z - Radius < -ZRange)
		{
			cout << "球和边界碰撞" << endl;
			CurrentPlace.z = -ZRange + Radius;
			CurrentSpeed.z = -CurrentSpeed.z;
		}
		else if (CurrentPlace.z + Radius > ZRange)
		{
			cout << "球和边界碰撞" << endl;
			CurrentPlace.z = ZRange - Radius;
			CurrentSpeed.z = -CurrentSpeed.z;
		}
	}

	/*
	描述：处理与球相撞，弹性碰撞
	返回：无
	*/
	void HandleCollisionBall(MovingBall& b)
	{
		Point diff = b.CurrentPlace - CurrentPlace;
		float dist = sqrt(diff * diff);
		if (dist < Radius + b.Radius)
		{
			cout << "球和球碰撞" << endl;

			//径向交换速度，法向速度不变
			Point speed_collide_self = diff * (CurrentSpeed * diff / dist / dist);
			Point speed_collide_b = diff * (b.CurrentSpeed * diff / dist / dist);
			Point unchanged_self = CurrentSpeed - speed_collide_self;
			Point unchanged_b = b.CurrentSpeed - speed_collide_b;
			Point new_self = unchanged_self + speed_collide_b;
			Point new_b = unchanged_b + speed_collide_self;
			CurrentSpeed = new_self;
			b.CurrentSpeed = new_b;
		}
	}
};