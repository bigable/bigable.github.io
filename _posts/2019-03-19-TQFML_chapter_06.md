---
layout: post
title: 《百面机器学习》第六章 概率图模型 阅读笔记
category: notes
tag: github
---

**01. 概率图模型的联合概率分布**  

---  

**问题1. 贝叶斯网络的联合概率分布**  

在给定A的条件下，B和C是条件独立的，基于条件概率的定义可得  

\begin{equation}  
P(C|A,B)= P(C|A)  
\end{equation}  

同理，在给定B和C的条件下，A和D是条件独立的，可得  

\begin{equation}  
P(D|A,B,C)= P(D|C,A)  
\end{equation}  

由上两式可得  

\begin{equation}  
P(A,B,C,D)= P(A)P(B|A)P(C|A)P(D|B,C)  
\end{equation}  


**问题2. 马尔科夫网络的联合概率分布**  
