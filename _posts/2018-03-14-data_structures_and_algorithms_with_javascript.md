---
layout: post
title: 《数据结构与算法JavaScript描述》学习笔记 (未完待续)
category: programming
---

![cover](https://raw.githubusercontent.com/bigablecat/data_structures_and_algorithms_using_javascript/master/cover.jpg)  

〇. 本项目地址：  
[https://github.com/bigablecat/javascript-datastructures-algorithms.git](https://github.com/bigablecat/javascript-datastructures-algorithms.git)

一. 官方随书代码下载:  
[https://github.com/oreillymedia/data_structures_and_algorithms_using_javascript](https://github.com/oreillymedia/data_structures_and_algorithms_using_javascript)

二. js shell安装和使用方法（以windows 10操作系统为例）：
1. 安装：下载mozilla官方的jsshell压缩包即可  
	1) 前往[https://archive.mozilla.org/pub/firefox/nightly/latest-mozilla-central/](https://archive.mozilla.org/pub/firefox/nightly/latest-mozilla-central/)  
	2) 找到自己的操作系统对应的jsshell压缩包，本例中下载的是[jsshell-win64.zip](https://archive.mozilla.org/pub/firefox/nightly/latest-mozilla-central/jsshell-win64.zip)  
	3) 解压缩jsshell-win64.zip文件到自己喜欢的路径，本例中最终解压路径为D:\jsshell-win64\  

2. 使用jsshell中的js.exe运行js文件：  
	新建一个hello.js文件，写一行代码print("hello world!")，有两种方式使用jsshell运行hello.js文件：  

	2.1) 方法一：在jsshell-win64文件夹下直接运行  
	2.1.1) 将hello.js文件放入jsshell-win64文件夹  
	2.1.2) 在jsshell-win64文件夹下，打开windows自带的cmd或PowerShell命令行工具  
	2.1.3) 在命令行输入.\js.exe .\hello.js  
	D:\jsshell-win64>.\js.exe .\hello.js  
	运行结果：hello world!  
	备注：js.exe和hello.js前需要加上.\，或者按tab键自动补全也会加上.\

	2.2) 方法二：将jsshell-win64/js.exe添加到windows系统环境变量，全局任意调用js.exe  
	2.2.1) 将jsshell文件夹添加到windows系统环境变量PATH  
	2.2.2) 进入hello.js所在文件夹，打开windows自带的cmd或PowerShell命令行工具  
	2.2.3) 在命令行输入js.exe .\hello.js  
	x:\xxx>js.exe .\hello.js  
	备注：hello.js前需要加上.\，或者按tab键自动补全也会加上.\