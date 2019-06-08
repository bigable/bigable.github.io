---
layout: post
title: 《机器学习》（周志华） 第3章 线性模型  
category: notes
tag: github
---

**3.1 基本形式**  
 
* 模型  
	>给定有 d 个属性描述的示例 $$x=(x_{1};x_{2};...;x_{d})$$  
	<br>其中 x<sub>i</sub> 是 x 在第 i 个属性上的取值  
	<br>线性模型（linear model）试图学得一个通过属性的线性组合来进行预测的函数
	<br>
	$$f(X)=w_{1}x_{1}+w_{2}x_{2}+...+w_{d}x_{d}+b$$
	<br>
	<br>用向量形式写成
	<br>
	<br>$$f(x)=\mathbf{w}^{T}\mathbf{x}+b$$
	<br>
	<br>其中 $$\mathbf{w}=(w_{1};w_{2};...;w_{d})$$
	<br>**w** 和 b 学得之后，模型即可确定

* 优点  

	>功能更为强大的非线性模型（nonlinear model）可在线性模型的基础上通过引入层级结构或高维映射而得
	
	>由于**w**直观表达了个属性在预测中的重要性
	<br>因此线性模型有很好的可解释性(comprehensibility)
	

**3.2 线性回归**  

*3.2.1 模型*  

>给定数据集 $$D=\{(\mathbf{x}_{1},y_{1}),(\mathbf{x}_{2},y_{2}),...,(\mathbf{x}_{m},y_{m})\}$$  
<br>其中 $$\mathbf{x}_{i}=(x_{i1};x_{i2};...;x_{id};),y_{i}\in\mathbb{R}$$  
<br>"线性回归"（linear regression）试图学得一个线性模型  
<br>以尽可能准确地预测实值输出标记  

*3.2.2 一元线性回归*  

* 假设输入属性只有一个，忽略关于属性的下标  

	>$$D=\{(x_{i},y_{i})\}_{i=1}^{m}$$  
	<br>
	<br>其中 $$x_{i}\in \mathbb{R}$$  

* 对离散属性  
	
	>若属性值间存在“序”（order）关系
	<br>可通过连续化将其转为连续值
	<br>若属性值间不存在序关系
	<br>假定有k个属性，通常转化为k维向量

* 线性回归试图学得  

	>$$f(x_{i})=wx_{i}+b$$，使得 $$f(x_{i})\simeq y_{i}$$


* 如何确定 w 和 b  
	
	>关键在于如何衡量 f(x) 与 y 之间的差别  

	- 均方误差  
		
		>回归任务中最常见的性能度量  
		<br>尝试让均方误差最小化  
		<br>
		<br>
		$$
		\begin{align*}
		(w^{*},b^{*}) 
		&=\underset{(w,b)}{\arg min}\sum_{m}^{i=1}(f(x_{i})-y_{i})^{2} \\
		&=\underset{(w,b)}{\arg min}\sum_{m}^{i=1}(y_{i} - wx_{i}-b)^{2}
		\end{align*}
		$$
	
		>均方误差有非常好的几何意义  
		<br>对应了常用的欧几里得距离或简称“欧氏距离”（Euclidien distance）  
	
	- 最小二乘法  

		>基于均方误差最小化来进行模型求解的方法称为“最小二乘法”（least square method）  
		<br>在线性回归中，最小二乘法就是试图找到一条直线
		<br>使所有样本到直线上的欧氏距离之和最小
	
	- 参数估计  
		
		>求解 w 和 b  
		<br>使 $$E_{(w,b)}=\sum_{m}^{i=1}(y_{i}-wx_{i}-b)$$ 最小化的过程  
		<br>
		<br>称为线性回归模型的最小二乘“参数估计”（parameter estimation）  
		<br>
		<br>将 $$E_{(w,b)}$$ 分别对 w 和 b 求导，得到  
		<br>
		<br>
		$$
		\frac{\partial E_{(w,b)}}{\partial w}
		=2(w\sum_{i=1}^{m}x_{i}^{2}-\sum_{i=1}^{m}(y_{i}-b)x_{i})
		$$
		<br>
		<br>
		$$
		\frac{\partial E_{(w,b)}}{\partial b}
		=2(mb-\sum_{i=1}^{m}(y_{i}-wx_{i}))
		$$  
		<br>
		<br>令 $$\frac{\partial E_{(w,b)}}{\partial w}=0$$ ， 
		$$\frac{\partial E_{(w,b)}}{\partial b}=0$$  
		<br>
		<br>得到 w 和 b 的最优解的闭式（close-form）解  
		<br>
		<br>
		$$
		w=\frac{\sum_{i=1}^{m}y_{i}(x_{i}-\bar{x})}
		{\sum_{i=1}^{m}x_{i}^{2}-\frac{1}{m}(\sum_{i=1}^{m}x_{i})^{2}}
		$$
		<br>
		<br>
		<br>
		$$b={\frac{1}{m}(\sum_{i=1}^{m}(y_{i}-wx_{i})}$$
		<br>
		<br>其中 $$\bar{x}=\frac{1}{m}\sum_{i=1}^{m}x_{i}$$
		<br>
		
*3.2.2 多元线性回归*  

* 样本有 d 个属性描述  

	>
	$$
	f(\mathbf{x_{i}})=
	\mathbf{w}^T \mathbf{x}_{i}+b
	$$，
	使得 $$f(\mathbf{x_{i}})\simeq y_{i}$$
	
* 最小二乘法估计 w 和 b  

	>把 w 和 b 吸收入向量形式 $$\hat{\mathbf{w}}=(\mathbf{w}^{T};b)$$  
	<br>把数据集D表示为一个 $$m \times (d+1)$$ 大小的矩阵 $$\mathbf{X}$$  
	<br>其中每行对应一个示例  
	<br>该行前d个元素对应于示例的 d 个属性值  
	<br>最后一个元素恒置为 1 
	<br>
	<br>
	$$
	\mathbf{X}=
	\begin{pmatrix}
    x_{11}& x_{12}& \cdots& x_{1d}& 1\\ 
    x_{21}& x_{22}& \cdots& x_{2d}& 1\\ 
    \vdots& \vdots& \ddots& \vdots& \vdots\\ 
    x_{m1}& x_{m2}& \cdots& x_{md}& 1\\ 
	\end{pmatrix}
	=
	\begin{pmatrix}
	x_{1}^{T}& 1\\
	x_{2}^{T}& 1\\
	\vdots& \vdots\\
	x_{m}^{T}& 1
	\end{pmatrix}
	$$
	<br>
	<br>
	<br>再把标记也写成向量形式 $$\mathbf{y}=(y_{1};y_{2};...;y_{m})$$  
	<br>
	<br>可以得到  
	<br>
	$$
	\def\*#1{\mathbf{#1}}
	\hat{\*{w}}^{*}=
	\underset{\hat{\*{w}}}{\arg min}
	(\*{y}-\*{X}\hat{\*{w}})^T
	(\*{y}-\*{X}\hat{\*{w}})
	$$
	<br>
	<br>令
	$$
	E_{\hat{\*{w}}}=
	\underset{\hat{\*{w}}}{\arg min}
	(\*{y}-\*{X}\hat{\*{w}})^T
	(\*{y}-\*{X}\hat{\*{w}})
	$$
	<br>
	<br>对 $$\hat{\mathbf{w}}$$求导得到
	<br>
	<br>
	$$
	\def\*#1{\mathbf{#1}}
	\frac{\partial E_{\hat{\*{w}}}}
	{\partial \hat{\*{w}}}=
	2\*{X}^T(\*{X}\hat{\*w}-\*{y})
	$$
	<br>
	