#pragma once


#include<math.h>
using namespace std;

#define PI 3.1415926

float max(float a, float b)
{
	if (a > b) return a;
	return b;
}

float min(float a, float b)
{
	if (a < b) return a;
	return b;
}

//3D点的基类
class Point
{
public:
	float x;
	float y;
	float z;
	Point()
	{
		x = 0;
		y = 0;
		z = 0;
	}
	Point(float tx, float ty, float tz)
	{
		x = tx;
		y = ty;
		z = tz;
	}
	void SetPlace(float tx, float ty, float tz)
	{
		x = tx;
		y = ty;
		z = tz;
	}
	Point operator+(const Point& b)
	{
		Point c;
		c.x = x + b.x;
		c.y = y + b.y;
		c.z = z + b.z;
		return c;
	}
	Point operator-(const Point& b)
	{
		Point c;
		c.x = x - b.x;
		c.y = y - b.y;
		c.z = z - b.z;
		return c;
	}
	Point operator*(const float& b)
	{
		Point c;
		c.x = x * b;
		c.y = y * b;
		c.z = z * b;
		return c;
	}
	float operator*(const Point& b)
	{
		float sum = 0;
		sum += x * b.x;
		sum += y * b.y;
		sum += z * b.z;
		return sum;
	}
};

