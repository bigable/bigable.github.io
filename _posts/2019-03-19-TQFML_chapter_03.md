---
layout: post
title: 《百面机器学习》第三章 经典算法 阅读笔记
category: notes
tag: github
---

## 02. 逻辑回归  

- 对数几率：  
	
	>逻辑回归公式整理可得：  
	<br>
	<br>$$log\frac{P}{1-P}=θ^{T}x $$  
	<br>
	<br>其中 $$p=P(y=1|x)$$  
	<br>
	<br>将给定输入 x 预测为正样本的概率  
	<br>如果把一个事件的几率(odds)定义为该事件发生的概率与不发生的概率的比值$$\frac{P}{1-P}$$  
	<br>那么逻辑回归可以看做是对 "y=1|x" 这一事件的对数几率的线性回归  

- 因变量：  
	
	>y 是因变量  
	<br>$$\frac{P}{1-P}$$不是因变量  
	<br>且自变量 x 与超参数 θ 确定的情况下  
	<br>逻辑回归可以看作广义线性模型（Generalized Linear Models）  
	<br>在因变量 y 服从二元分布时的一个特殊情况  
	<br>使用最小二乘法求解线性回归时，因变量 y 服从正态分布  


**问题1. 逻辑回归和线性回归的异同**  

**1. 不同**  

* 本质区别  

	>逻辑回归处理分类问题  
	<br>线性回归处理回归问题  

* 模型不同  

	- 逻辑回归：  
		
		>因变量取值是一个二元分布  
		<br>模型学习得出 $$E[y|x;θ]$$  
		<br>即给定自变量和超参数后，得到因变量的期望  
		<br>并基于此期望来处理预测分类问题  
		
	- 线性回归：  
		
		>求解 $$y^{'}=θ^{T}x$$  
		<br>是对真实关系 $$y^{'}=θ^{T}x+\epsilon$$ 的一个近似  
		<br>其中 $$\epsilon$$ 代表误差项  
		<br>我们使用近似项来处理回归问题  

* 因变量不同  

	- 逻辑回归：  
		
		>因变量是离散的  

	- 线性回归：  
		
		>因变量是连续的  

**2. 相同**  

* 都使用了极大似然估计对训练样本建模  
	
	- 线性回归：  

		>使用最小二乘法  
		<br>实际上就是在自变量 x 与超参数 θ 确定  
		<br>因变量 y 服从正态分布的假设下  
		<br>使用极大似然估计的一个化简  
		
	- 逻辑回归：  
		
		>通过对似然函数  
		<br>
		<br>$$L(θ)=\prod_{N}^{i=1} P(y_{i}|x_{i};\theta)=\prod_{N}^{i=1} (\pi (x_{i}))^{y_{i}}(1-\pi(x_{i}))^{1-y_{i}}$$  
		<br>
		<br>的学习，得到最佳参数 θ  

* 求解超参数的过程中，都可以使用梯度下降的方法  

**问题2. 逻辑回归处理多标签分类问题的常见做法和应用场景**  

*2.1 一个样本对应**一个**标签的情况*  

* 多项逻辑回归（SoftMax Regression）  
	>假设每个样本属于不同表情的概率服从于集合分布  
	<br>使用多项逻辑回归进行分类：  
	<br>
	<br>
$$
h_{\theta}=
\begin{bmatrix}
p(y=1|x;\theta)\\ 
p(y=2|x;\theta)\\ 
\vdots \\
p(y=k|x;\theta)
\end{bmatrix}
=\frac{1}{\sum_{k}^{j=1}e^{\theta_{j}^{T}x}}
\begin{bmatrix} 
e^{\theta_{1}^{T}x}\\
e^{\theta_{2}^{T}x}\\
\vdots\\
e^{\theta_{k}^{T}x}\\
\end{bmatrix}
$$  
	<br>
	<br>其中 $$\theta_{1},\theta_{2},...,\theta_{k} \in \mathbb{R}$$ 为模型的参数  
	<br>$$\frac{1}{\sum_{k}^{j=1}e^{\theta_{j}^{T}x}}$$ 可以看作对概率的归一化  
	<br>为方便起见，将$${\theta_{1},\theta_{2},...,\theta_{k}}$$ 这k个向量按顺序排列形成n×k维矩阵  
	<br>写作 θ ，表示整个参数集  
	
* 参数冗余：  

	>一般来说，多项逻辑回归有参数冗余的特点  
	<br>将 $${\theta_{1},\theta_{2},...,\theta_{k}}$$ 同时加减一个向量后预测结果不变  
	
* 当参数类别为2时：  
	
	>  
$$
h_{\theta}(x)=\frac{1}{e^{\theta_{1}^{T}x}+e^{\theta_{2}^{T}x}}
\begin{bmatrix}
e^{\theta_{1}^{T}x}\\
e^{\theta_{2}^{T}x}
\end{bmatrix}
$$  
	<br>
	<br>利用参数冗余的特点，将所有参数减去 $$\theta_{1}$$  
	<br>
	<br>
$$
\begin{align*}
h_{\theta}(x)
&=
\frac{1}{e^{0\cdot x}+e^{(\theta_{2}^{T}-\theta_{1}^{T})x}}
\begin{bmatrix}
e^{0\cdot x}\\
e^{(\theta_{2}^{T}-\theta_{1}^{T})x}
\end{bmatrix}
\\
&=
\begin{bmatrix}
\frac{1}{1+e^{\theta^{T}x}}\\
1-\frac{1}{1+e^{\theta^{T}x}}
\end{bmatrix}
\end{align*}
$$
	<br>
	<br>其中 $$\theta=\theta_{2}-\theta_{1}$$  
	<br>整理后的式子和逻辑回归一致  
	<br>多项逻辑回归实际上是二分类逻辑回归在多标签下的一种拓展  

*2.2 一个样本属于**多个**标签的情况*  

>训练 k 个二分类的逻辑回归分类器  
<br>训练第 i 个分类器时，将标签重新整理为
<br>“第 i 类标签”  
<br>“非第 i 类标签”  
<br>用这种方法区分每个样本是否可以归为第 i 类  
