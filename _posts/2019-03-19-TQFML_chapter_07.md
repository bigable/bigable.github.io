---
layout: post
title: 《百面机器学习》第七章 优化算法 阅读笔记
category: notes
tag: github
---

**01. 有监督学习的损失函数**  

---  

**问题1. 有监督学习涉及的损失函数有哪些**  

* 损失函数  

	假设训练样本的形式为  

	>(x<sub>i</sub>,y<sub>i</sub>)  
<br>其中  
<br>x<sub>i</sub>∈X，表示第i个样本点的特征  
<br>y<sub>i</sub>∈Y，表示该样本点的标签  
<br>参数为θ的模型可以表示为函数 f(·,θ):X → Y  
<br>模型关于第i个样本点的输出为 f(x<sub>i</sub>,θ)  

	在有监督学习中，损失函数刻画了模型和训练样本的匹配程度  

	为了刻画模型输出样本与样本标签的匹配程度，定义损失函数  

	>L(·,·):Y×Y→$\mathbb{R}$<sub>≥0</sub>  
<br>L(f(x<sub>i</sub>,θ),y<sub>i</sub>)越小，表明模型在该样本点匹配得越好  

	对二分类问题  

	>Y={1,-1}，我们希望sign f(x<sub>i</sub>,θ)=y  
<br>最自然的损失函数是0-1损失，即 L<sub>0-1</sub>(f,y)=l<sub>fy≤0</sub>  
<br>其中  
<br>l<sub>P</sub>是指示函数（Indicator Function）  
<br>当且仅当P为真时取值为1，否则取值为0  
<br>0-1损失函数能够直观地刻画分类的错误率  
<br>由于其非凸、非光滑的特点，使得算法很难直接对函数进行优化  

* 0-1损失函数的一个代理损失函数是**Hinge**损失函数：  

	>L<sub>hinge</sub>(f,y) = max{0,1 - fy}  
<br>Hinge损失函数是0-1损失函数相对紧的凸上界  
<br>且当fy≥1时，该函数不对其做任何惩罚  
<br>Hinge损失在fy=1处不可导，因此不能用梯度下降法优化  
<br>Hinge损失函数使用次梯度下降法（Subgradient Descent Method）优化  

* 0-1损失的另一个代理损失函数是**Logistic**损失函数  

	>L<sub>logistic</sub>(f,y)=log<sub>2</sub>(1+exp(-fy))  
<br>Logistic损失函数也是0-1损失函数的凸上界  
<br>Logistic损失函数处处光滑，可以使用梯度下降法优化  
<br>Logistic损失函数对所有样本点都有所惩罚，因此对异常值相对敏感  

* 当预测值f∈[-1,1]时，另一个常用的代理损失函数是**交叉熵（Cross Entropy）**损失函数

	>L<sub>cross entropy</sub>(f,y)=-log<sub>2</sub>($\frac{1+fy}{2}$)  
<br>交叉熵损失函数也是0-1损失函数的光滑凸上界  
