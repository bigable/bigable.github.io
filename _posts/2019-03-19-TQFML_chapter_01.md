---
layout: post
title: 《百面机器学习》第一章 特征工程 阅读笔记
category: notes
tag: github
---

## 01. 特征归一化  

>为了消除数据特征之间的量纲影响  
<br>需要对数据进行归一化（Normalization）处理  
<br>使各指标处于同一数量级，以便进行分析  

**问题1. 为什么需要对数值类型进行归一化处理**  

**1.1 为什么归一化？**  

* 使不同指标具有可比性  

	>对数值类型进行归一化可以将所有特征都统一到一个大致相同的数值区间内  
	<br>指标处于同一数值量级，便于分析  

* 梯度下降中使用归一化  

	>在梯度下降中，归一化之后不同特征的更新速度更为一致  

* **适用**归一化的模型  

	>通过梯度下降的模型通常是需要归一化的  
	<br>如线性回归，逻辑回归，支持向量机，神经网络  

- **不适**用归一化的模型  

	>决策树模型  
	
	- C4.5为例  

	  >决策树在进行节点分裂时主要依据数据集D关于特征x的信息增益比  
	  <br>归一化并不会改变样本在特征x上的信息增益  


**1.2 常用的归一化方法**  

* 线性函数归一化（Min-Max Scaling）：  

	>对原始数据进行线性变换  
	<br>使结果映射到[0,1]的范围  
	<br>实现对原始数据的等比缩放  

	>公式：  
	<br>$$X_{d}=\frac{X-X_{min}}{X_{max}-X_{min}}$$  
	<br>
	<br>X为原始数据，X<sub>max</sub>、X<sub>min</sub>分别为最大值和最小值  

* 零均值归一化（Z-Score Normalization）：  

	>将原始数据映射到均值为0、标准差为1的分布上  

	>公式：  
	<br>$$z=\frac{x-μ}{σ}$$  

##  

## 02. 类别型特征  

>类别型特征（Categorical Feature）主要指性别、血型等只在有限选项内取值的特征  
<br>类别型特征原始输入通常是字符串形式  
<br>除了决策树等少数模型能直接处理字符串形式的输入  
<br>对于逻辑回归、支持向量机等模型  
<br>类别型特征必须转换成数值型特征  

**问题1. 数据预处理时，怎样处理类别型特征**  

