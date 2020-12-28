#include<freeglut/glut.h>
#include<iostream>
#include<math.h>
#include<windows.h>
#include "Point.hpp"
#include "MovingBall.hpp"
#include "Light.hpp"
#include "Camera.hpp"
#include "Board.hpp"
using namespace std;


//全局常量
const int WindowSizeX = 800, WindowSizeY = 600, WindowPlaceX = 100, WindowPlaceY = 100;
const char WindowName[] = "MyScene";
const float TimeOnce = 0.02; //刷新时间
const float XRange = 10, ZRange = 10, Height = 8; //场景的X,Y,Z范围（-X,X),(0,H),(-Z,Z)
const int BallComplexity = 40; //小球绘制精细程度

//光照，相机
Camera TheCamera;
Light TheLight;

//物体
Board Boards[5]; //边界
//小球
MovingBall BallA; 
MovingBall BallB;




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

	//地板材质
	GLfloat color_floor[3] = { 1.0, 1.0, 1.0 };
	GLfloat ambient_floor[3] = { 0.4, 0.4, 0.4};
	GLfloat diffuse_floor[3] = { 0.2, 0.2, 0.2};
	GLfloat specular_floor[3] = { 0.4, 0.4, 0.4};
	GLfloat shininess_floor = 90;
	Boards[0].InitColor(color_floor, ambient_floor, diffuse_floor, specular_floor, shininess_floor);

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


//初始化小球
void InitMovingBalls()
{
	//小球A的位置，速度
	float radius_a = 1;
	Point place_a = Point(3, 0, -3);
	Point speed_a = Point(10, 0, -6);

	//小球A的纹理，材质，颜色
	GLfloat color_a[3] = { 1.0, 0.0, 0.0 };
	GLfloat ambient_a[3] = { 0.4, 0.2, 0.2 };
	GLfloat diffuse_a[3] = { 1, 0.8, 0.8 };
	GLfloat specular_a[3] = { 0.5, 0.3, 0.3 };
	GLfloat shininess_a = 10;

	//初始化小球A
	BallA.InitPlace(place_a.x, place_a.z, radius_a, speed_a.x, speed_a.z);
	BallA.InitColor(color_a, ambient_a, diffuse_a, specular_a, shininess_a);

	//小球B的位置，速度
	float radius_b = 1;
	Point place_b = Point(-3, 0, -3);
	Point speed_b = Point(7, 0, 10);

	//小球B的纹理，材质，颜色
	GLfloat color_b[3] = { 0.0, 0.0, 1.0 };
	GLfloat ambient_b[3] = { 0.2, 0.2, 0.4 };
	GLfloat diffuse_b[3] = { 0.3, 0.3, 0.6 };
	GLfloat specular_b[3] = { 0.8, 0.8, 1.0 };
	GLfloat shininess_b = 80;

	//初始化小球B
	BallB.InitPlace(place_b.x, place_b.z, radius_b, speed_b.x, speed_b.z);
	BallB.InitColor(color_b, ambient_b, diffuse_b, specular_b, shininess_b);
}

//初始化的主函数
void InitScene()
{

	InitLight();
	InitCamera();
	InitBoards();
	InitMovingBalls();
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


//进行小球位置更新和碰撞检测，处理
void UpdateBalls()
{
	BallA.Move(TimeOnce);
	BallA.HandleCollisionBoard(XRange, ZRange);
	BallB.Move(TimeOnce);
	BallB.HandleCollisionBoard(XRange, ZRange);
	BallA.HandleCollisionBall(BallB);
	
}

//绘制一个小球
void DrawOneBall(MovingBall& ball)
{
	//设置纹理，材质等信息
	glMaterialfv(GL_FRONT, GL_AMBIENT, ball.Ambient);
	glMaterialfv(GL_FRONT, GL_DIFFUSE, ball.Diffuse);
	glMaterialfv(GL_FRONT, GL_SPECULAR, ball.Specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, ball.Shininess);

	//平移到坐标原点，绘制，恢复坐标
	glPushMatrix();
	glTranslatef(ball.CurrentPlace.x, ball.CurrentPlace.y, ball.CurrentPlace.z);
	glutSolidSphere(ball.Radius, BallComplexity, BallComplexity);
	glPopMatrix();
}

//绘制小球
void DrawBalls()
{
	//更新小球位置
	UpdateBalls();
	
	//绘制小球
	DrawOneBall(BallA);
	DrawOneBall(BallB);
}

//绘制的主函数
void DrawScene()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);	//清除颜色缓存
	SetCamera();//设置相机
	DrawBoards();//绘制地板和边框
	DrawBalls();//更新和绘制小球
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

int main(int argc, char**argv)
{

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
