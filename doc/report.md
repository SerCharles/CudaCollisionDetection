# 基于CUDA的碰撞检测报告

软73 沈冠霖 2017013569



## 1.运行环境与方法

#### 1.1 运行环境

- 系统：Win10
- 开发环境：VS2017
- OpenGL版本：4.6.0 NVIDIA 441.66
- OGLU工具库版本：1.2.2.0 Microsoft Corporation
- CUDA版本：10.2
- 使用GPU device 0: GeForce GTX 950M
- SM的数量：5
- 每个线程块的共享内存大小：48 KB
- 每个线程块的最大线程数：1024
- 每个EM的最大线程数：2048
- 每个EM的最大线程束数：64

#### 1.2 运行方法

打开bin中的可执行文件，选择4种模式，能看到动画渲染效果，并且运行几分钟后能看到一万次碰撞检测平均时间。（模式2，也就是串行网格碰撞检测较慢，不建议打开）



## 2.实现原理

#### 2.1 无优化的碰撞检测

//逐个匹配

//动量守恒公式



#### 2.2 CUDA并行化的无优化碰撞检测

//简单描述改了啥就行



#### 2.3 基于网格的碰撞检测

//网格划分，home和phantom

//基数排序

//碰撞检测与判重



#### 2.4 CUDA并行化的基于网格的碰撞检测

//cuda如何进行基数排序

//cuda如何进行碰撞检测和判重



## 3.实验结果

#### 3.1 正确性

//基本正确

//渲染的时候有时候物体会陷进去，小问题

#### 3.2 效率比较

//放图



## 4.总结

#### 4.1 总结



#### 4.2 参考文献

[1]CUDA的配置按照这个教程 https://blog.csdn.net/u013165921/article/details/77891913

[2]CUDA学习按照这个教程 https://zhuanlan.zhihu.com/p/34587739

[3]算法基本按照这个流程 https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-32-broad-phase-collision-detection-cuda

[4]代码实现参考了这个repo https://github.com/deeptoaster/cuda-collisions