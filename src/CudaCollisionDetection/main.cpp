#include<freeglut/glut.h>
#include<iostream>
#include<math.h>
#include<windows.h>
#include "Point.hpp"
#include "Ball.hpp"
#include "Light.hpp"
#include "Camera.hpp"
#include "Board.hpp"
#include "BallList.hpp"
#include "collision.cuh"
using namespace std;


//全局常量
const int WindowSizeX = 800, WindowSizeY = 600, WindowPlaceX = 100, WindowPlaceY = 100;
const char WindowName[] = "MyScene";
const float TimeOnce = 0.02; //刷新时间
const int BallNum = 4;
const float XRange = 18, ZRange = 18, Height = 36, MaxRadius = 1; //场景的X,Y,Z范围（-X,X),(0,H),(-Z,Z)
int GlobalMode = -1;

//光照，相机
Camera TheCamera;
Light TheLight;

//物体
Board Boards[6]; //边界

BallList Balls;




//初始化函数集合
//初始化窗口
void InitWindow()
{
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	glutInitWindowSize(WindowSizeX, WindowSizeY);
	glutInitWindowPosition(WindowPlaceX, WindowPlaceY);
	glutCreateWindow(WindowName);
	const GLubyte* OpenGLVersion = glGetString(GL_VERSION);
	const GLubyte* gluVersion = gluGetString(GLU_VERSION);
	printf("OpenGL实现的版本号：%s\n", OpenGLVersion);
	printf("OGLU工具库版本：%s\n", gluVersion);
	int dev = 0;
	cudaDeviceProp devProp;
	if (cudaGetDeviceProperties(&devProp, dev) == cudaSuccess)
	{
		std::cout << "使用GPU device " << dev << ": " << devProp.name << std::endl;
		std::cout << "SM的数量：" << devProp.multiProcessorCount << std::endl;
		std::cout << "每个线程块的共享内存大小：" << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
		std::cout << "每个线程块的最大线程数：" << devProp.maxThreadsPerBlock << std::endl;
		std::cout << "每个EM的最大线程数：" << devProp.maxThreadsPerMultiProcessor << std::endl;
		std::cout << "每个EM的最大线程束数：" << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;
	}
}

//初始化光照
void InitLight()
{
	GLfloat background_color[3] = { 0.0, 0.0, 0.0 };
	GLfloat ambient[3] = { 1, 1, 1};
	GLfloat diffuse[3] = { 1, 1, 1};
	GLfloat specular[3] = { 1, 1, 1};
	GLfloat position[3] = { 0.0f, 10.0f, 0.0f};
	TheLight.Init(background_color, ambient, diffuse, specular, position);

	//设置着色模式
	glShadeModel(GL_SMOOTH);
	//设置初始背景色，清除颜色缓存和深度缓存
	glClearColor(TheLight.Color[0], TheLight.Color[1], TheLight.Color[2], TheLight.Color[3]);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//设置光源信息
	glLightfv(GL_LIGHT0, GL_AMBIENT, TheLight.Ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, TheLight.Diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, TheLight.Specular);
	glLightfv(GL_LIGHT0, GL_POSITION, TheLight.Position);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	
	//设置深度检测，即只绘制最前面的一层
	glEnable(GL_DEPTH_TEST);
}

//初始化相机
void InitCamera()
{
	//设置初始相机位置
	TheCamera.Init(10.0f, 10.0f);
}

//初始化边界和地板
void InitBoards()
{
	//8个点
	Point DownA(-XRange, 0, -ZRange);
	Point DownB(-XRange, 0, ZRange);
	Point DownC(XRange, 0, -ZRange);
	Point DownD(XRange, 0, ZRange);
	Point UpA(-XRange, Height, -ZRange);
	Point UpB(-XRange, Height, ZRange);
	Point UpC(XRange, Height, -ZRange);
	Point UpD(XRange, Height, ZRange);

	//设置地板和挡板位置
	Boards[0].InitPlace(DownA, DownB, DownD, DownC);
	Boards[1].InitPlace(DownA, DownB, UpB, UpA);
	Boards[2].InitPlace(DownC, DownD, UpD, UpC);
	Boards[3].InitPlace(DownA, DownC, UpC, UpA);
	Boards[4].InitPlace(DownB, DownD, UpD, UpB);
	Boards[5].InitPlace(UpA, UpB, UpD, UpC);

	GLfloat color_down[3] = { 1.0, 1.0, 1.0 };
	GLfloat ambient_down[3] = { 0.4, 0.4, 0.4 };
	GLfloat diffuse_down[3] = { 0.4, 0.4, 0.4 };
	GLfloat specular_down[3] = { 0.2, 0.2, 0.2 };
	GLfloat shininess_down = 20;
	Boards[0].InitColor(color_down, ambient_down, diffuse_down, specular_down, shininess_down);

	//设置四周挡板材质
	GLfloat color_border[3] = { 1.0, 1.0, 1.0 };
	GLfloat ambient_border[3] = { 0.2, 0.2, 0.2};
	GLfloat diffuse_border[3] = { 0.2, 0.2, 0.2};
	GLfloat specular_border[3] = { 0.2, 0.2, 0.2};
	GLfloat shininess_border = 40;
	for (int i = 1; i < 5; i++)
	{
		Boards[i].InitColor(color_border, ambient_border, diffuse_border, specular_border, shininess_border);
	}
}




//初始化的主函数
void InitScene()
{

	InitLight();
	InitCamera();
	InitBoards();
	Balls.Init(XRange, Height, ZRange, BallNum, MaxRadius, TimeOnce, GlobalMode);
	Balls.InitBalls();
}

//绘制函数集合
//设置相机位置
void SetCamera()
{
	glLoadIdentity();
	Point camera_place = TheCamera.CurrentPlace;//这就是视点的坐标  
	Point camera_center = TheCamera.LookCenter;//这是视点中心坐标
	gluLookAt(camera_place.x, camera_place.y, camera_place.z, camera_center.x, camera_center.y, camera_center.z, 0, 1, 0); //从视点看远点,y轴方向(0,1,0)是上方向  
}

//绘制边界和地板
void DrawBoards()
{
	for (int i = 0; i < 5; i++)
	{
		glColor3f(Boards[i].Color[0], Boards[i].Color[1], Boards[i].Color[2]);
		glMaterialfv(GL_FRONT, GL_AMBIENT, Boards[i].Ambient);
		glMaterialfv(GL_FRONT, GL_DIFFUSE, Boards[i].Diffuse);
		glMaterialfv(GL_FRONT, GL_SPECULAR, Boards[i].Specular);
		glMaterialfv(GL_FRONT, GL_SHININESS, Boards[i].Shininess);


		glBegin(GL_POLYGON);
		glVertex3f(Boards[i].PointList[0].x, Boards[i].PointList[0].y, Boards[i].PointList[0].z);
		glVertex3f(Boards[i].PointList[1].x, Boards[i].PointList[1].y, Boards[i].PointList[1].z);
		glVertex3f(Boards[i].PointList[2].x, Boards[i].PointList[2].y, Boards[i].PointList[2].z);
		glVertex3f(Boards[i].PointList[3].x, Boards[i].PointList[3].y, Boards[i].PointList[3].z);
		glEnd();
		glFlush();
	}
	
}


//绘制的主函数
void DrawScene()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);	//清除颜色缓存
	SetCamera();//设置相机
	DrawBoards();//绘制地板和边框
	Balls.UpdateBalls();
	Balls.DrawBalls();//更新和绘制小球
	glutSwapBuffers();
}

//全局定时器
void OnTimer(int value)
{
	glutPostRedisplay();//标记当前窗口需要重新绘制，调用myDisplay()
	glutTimerFunc(20, OnTimer, 1);
}


//交互函数集合
//处理鼠标点击 
void OnMouseClick(int button, int state, int x, int y)  
{
	if (state == GLUT_DOWN)
	{
		TheCamera.MouseDown(x, y);
	}
}

//处理鼠标拖动  
void OnMouseMove(int x, int y) 
{
	TheCamera.MouseMove(x, y);
}

//处理键盘点击（WASD）
void OnKeyClick(unsigned char key, int x, int y)
{
	int type = -1;
	if (key == 'w')
	{
		type = 0;
	}
	else if (key == 'a')
	{
		type = 1;
	}
	else if (key == 's')
	{
		type = 2;
	}
	else if (key == 'd')
	{
		type = 3;
	}
	TheCamera.KeyboardMove(type);
}

//处理键盘点击（前后左右）
void OnSpecialKeyClick(GLint key, GLint x, GLint y)
{
	int type = -1;
	if (key == GLUT_KEY_UP)
	{
		type = 0;
	}
	if (key == GLUT_KEY_LEFT)
	{
		type = 1;
	}
	if (key == GLUT_KEY_DOWN)
	{
		type = 2;
	}
	if (key == GLUT_KEY_RIGHT)
	{
		type = 3;
	}
	TheCamera.KeyboardMove(type);
}


//reshape函数
void reshape(int w, int h)
{
	glViewport(0, 0, w, h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(75.0f, (float)w / h, 1.0f, 1000.0f);
	glMatrixMode(GL_MODELVIEW);
}

void InitSettings()
{
	while (1)
	{
		int mode;
		cout << "请输出碰撞检测算法类型" <<endl;
		cout << "0：串行，循环碰撞检测" << endl;
		cout << "1：并行，循环碰撞检测" << endl;
		cout << "2：串行，空间划分碰撞检测" << endl;
		cout << "3：并行，空间划分碰撞检测" << endl;
		cin >> mode;
		if (mode >= 0 && mode <= 3)
		{
			cout << "当前模式为：";
			if (mode == NAIVE_CPU)
			{
				cout << "0：串行，循环碰撞检测" << endl;
			}
			else if (mode == NAIVE_GPU)
			{
				cout << "1：并行，循环碰撞检测" << endl;
			}
			else if (mode == FAST_CPU)
			{
				cout << "2：串行，空间划分碰撞检测" << endl;
			}
			else if (mode == FAST_GPU)
			{
				cout << "3：并行，空间划分碰撞检测" << endl;
			}
			GlobalMode = mode;
			break;
		}
		else
		{
			printf("输入不合法，请重新输入！\n");
		}
	}
}

int main(int argc, char**argv)
{
	InitSettings();
	glutInit(&argc, argv); 
	InitWindow();             //初始化窗口
	InitScene();              //初始化场景
	glutReshapeFunc(reshape); //绑定reshape函数
	glutDisplayFunc(DrawScene); //绑定显示函数
	glutTimerFunc(20, OnTimer, 1);  //启动计时器
	glutMouseFunc(OnMouseClick); //绑定鼠标点击函数
	glutMotionFunc(OnMouseMove); //绑定鼠标移动函数
	glutKeyboardFunc(OnKeyClick);//绑定键盘点击函数
	glutSpecialFunc(OnSpecialKeyClick);//绑定特殊键盘点击函数
	glutMainLoop();
}
