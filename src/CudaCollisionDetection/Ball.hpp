#pragma once
#include<math.h>
#include<freeglut/glut.h>
#include<vector>
#include"Point.hpp"
#include"Board.hpp"
using namespace std;

#pragma once
class Ball
{
public:
	//λ�ã��ٶ���Ϣ
	Point CurrentPlace;
	float Radius;
	Point CurrentSpeed;
	int BallComplexity;
	float Weight;

	//���ʣ�������ɫ��Ϣ
	GLfloat Color[3] = { 0, 0, 0 }; //��ɫ
	GLfloat Ambient[4] = { 0, 0, 0, 0 }; //������
	GLfloat Diffuse[4] = { 0, 0, 0, 0 }; //������
	GLfloat Specular[4] = { 0, 0, 0, 0 }; //���淴��
	GLfloat Shininess[4] = { 0 }; //����ָ��
public:
	Ball(){}

	//��ʼ��λ�ã��ٶ���Ϣ
	void InitPlace(float x, float y, float z, float radius, float speed_x, float speed_y, float speed_z)
	{
		CurrentPlace.SetPlace(x, y, z);
		CurrentSpeed.SetPlace(speed_x, speed_y, speed_z);
		Radius = radius;
		Weight = radius * radius * radius;
	}

	//��ʼ����ɫ������������Ϣ
	void InitColor(GLfloat color[], GLfloat ambient[], GLfloat diffuse[], GLfloat specular[], GLfloat shininess, int complexity)
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
		BallComplexity = complexity;
	}

	//����һ��С��
	void DrawSelf()
	{
		//�����������ʵ���Ϣ
		glMaterialfv(GL_FRONT, GL_AMBIENT, Ambient);
		glMaterialfv(GL_FRONT, GL_DIFFUSE, Diffuse);
		glMaterialfv(GL_FRONT, GL_SPECULAR, Specular);
		glMaterialfv(GL_FRONT, GL_SHININESS, Shininess);

		//ƽ�Ƶ�����ԭ�㣬���ƣ��ָ�����
		glPushMatrix();
		glTranslatef(CurrentPlace.x, CurrentPlace.y, CurrentPlace.z);
		glutSolidSphere(Radius, BallComplexity, BallComplexity);
		glPopMatrix();
	}
};


class BallList
{
public:
	vector<Ball> balls;
	BallList() {}
	float XRange;
	float ZRange;
	float Height;
	int Num;
	float MaxRadius;

	void Init(float x, float y, float z, int num, float max_radius)
	{
		XRange = x;
		ZRange = z;
		Height = y;
		Num = num;
		MaxRadius = max_radius;
		
	}

	void InitBalls()
	{
		//С����������ʣ���ɫ
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
					Ball new_ball;
					new_ball.InitColor(color, ambient, diffuse, specular, shininess, complexity);
					float speed_x = (rand() % 201) / 100.0f - 1.0f;
					float speed_y = (rand() % 201) / 100.0f - 1.0f;
					float speed_z = (rand() % 201) / 100.0f - 1.0f;
					float radius = ((rand() % 51) / 100.0f + 0.5f) * MaxRadius;
					new_ball.InitPlace(place_x, place_y, place_z, radius, speed_x, speed_y, speed_z);
					balls.push_back(new_ball);
				}
			}

		}
	}

	//����������
	void DrawBalls()
	{
		for (int i = 0; i < balls.size(); i++)
		{
			balls[i].DrawSelf();
		}
	}

	//Cuda��������˶�
	void UpdateBalls()
	{
		//TODO
		//Cuda
	}

};