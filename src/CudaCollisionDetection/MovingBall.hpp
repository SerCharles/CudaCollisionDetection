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
	//λ�ã��ٶ���Ϣ
	Point CurrentPlace;
	float Radius;
	Point CurrentSpeed;

	//���ʣ�������ɫ��Ϣ
	GLfloat Color[3] = { 0, 0, 0 }; //��ɫ
	GLfloat Ambient[4] = { 0, 0, 0, 0 }; //������
	GLfloat Diffuse[4] = { 0, 0, 0, 0 }; //������
	GLfloat Specular[4] = { 0, 0, 0, 0 }; //���淴��
	GLfloat Shininess[4] = { 0 }; //����ָ��
public:
	MovingBall(){}

	//��ʼ��λ�ã��ٶ���Ϣ
	void InitPlace(float x, float z, float radius, float speed_x, float speed_z)
	{
		CurrentPlace.SetPlace(x, radius, z);
		CurrentSpeed.SetPlace(speed_x, 0, speed_z);
		Radius = radius;
	}

	//��ʼ����ɫ������������Ϣ
	void InitColor(GLfloat color[], GLfloat ambient[], GLfloat diffuse[], GLfloat specular[], GLfloat shininess)
	{
		for (int i = 0; i < 3; i++)
		{
			Color[i] = color[i];
			Ambient[i] = ambient[i];
			Diffuse[i] = diffuse[i];
			Specular[i] = specular[i];
		}
		//͸���ȣ�1
		Ambient[3] = 1.0;
		Diffuse[3] = 1.0;
		Specular[3] = 1.0;
		Shininess[0] = shininess;
	}

	//��㵽���ľ���
	float GetDistance(Point p)
	{ 
		return sqrt((p - CurrentPlace) * (p - CurrentPlace));
	}

	//�����ƶ�
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
		������������߽���ײ
		������X��Χ��-X,X),Z��Χ(-Z,Z)
		���أ���
	*/
	void HandleCollisionBoard(float XRange, float ZRange)
	{
		if (CurrentPlace.x - Radius < -XRange)
		{
			cout << "��ͱ߽���ײ" << endl;
			CurrentPlace.x = -XRange + Radius;
			CurrentSpeed.x = -CurrentSpeed.x;
		}
		else if (CurrentPlace.x + Radius > XRange)
		{
			cout << "��ͱ߽���ײ" << endl;
			CurrentPlace.x = XRange - Radius;
			CurrentSpeed.x = -CurrentSpeed.x;
		}
		if (CurrentPlace.z - Radius < -ZRange)
		{
			cout << "��ͱ߽���ײ" << endl;
			CurrentPlace.z = -ZRange + Radius;
			CurrentSpeed.z = -CurrentSpeed.z;
		}
		else if (CurrentPlace.z + Radius > ZRange)
		{
			cout << "��ͱ߽���ײ" << endl;
			CurrentPlace.z = ZRange - Radius;
			CurrentSpeed.z = -CurrentSpeed.z;
		}
	}

	/*
	����������������ײ��������ײ
	���أ���
	*/
	void HandleCollisionBall(MovingBall& b)
	{
		Point diff = b.CurrentPlace - CurrentPlace;
		float dist = sqrt(diff * diff);
		if (dist < Radius + b.Radius)
		{
			cout << "�������ײ" << endl;

			//���򽻻��ٶȣ������ٶȲ���
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