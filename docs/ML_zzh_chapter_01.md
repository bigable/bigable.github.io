**1.2 基本术语**  

* 模型（model）：  

	>本书中的“模型”泛指从数据中学得的结果  

* 数据集（data set）：  

	>数据记录的集合  

* 示例（instance）/样本（sample）：  

	>数据集中的每条记录是关于一个事件或对象的描述  
	
* 属性（attribute）：  
	
	>反映事件或对象在某方面的表现或性质的事项  
	<br>称为“属性”（attribute）或“特征”（feature）  

* 属性值（attribute value）：  
	
	>属性上的取值，称为属性值  

* 属性空间（attribute space）：  

	>属性张成的空间，称为属性空间，样本空间（sample space）或输入空间  
	
* 特征向量（feature vector）：  

	>如果将每个特征作为一个坐标轴，
	<br>这些坐标轴张成一个多维空间，
	<br>空间中每个点对应一个坐标向量，
	<br>因此一个示例也称为一个特征向量  

	>令 D={x<sub>1</sub>, x<sub>2</sub>,…,x<sub>m</sub>}
	<br>表示包含 m 个示例的数据集，
	<br>每个示例由 d 个属性描述，
	<br>则每个示例 x<sub>i</sub>={x<sub>i1</sub>;x<sub>i2</sub>;…;x<sub>id</sub>}  
	<br>是d维样本空间中的一个向量，
	<br>x<sub>i</sub>∈X，其中 x<sub>ij</sub> 是 x<sub>i</sub> 在第j个属性上的取值，
	<br> d 称为样本 x<sub>i</sub> 的“维数”（dimensionality）  

* 训练（training）：  

	>从数据中学得模型的过程称为“学习”（learning）或训练（training）  
	<br>这个过程用通过执行某个学习算法来完成
	
	>训练过程中使用的数据称为“训练数据”（training data）
	<br>其中每个样本称为一个“训练样本”（training sample）
	<br>训练样本组成的集合称为“训练集”（training set）
	
	>学得模型对应了关于数据的某种潜在的规律，因此亦称“假设（hypothesis）
	
	>这种潜在规律自身，则称为“真相”或“真实”（ground-truth）  
	<br>学习过程就是为了找出或逼近真相
	
	>本书有时将模型称为“学习器”（learner）  
	<br>可看做算法在给定数据或参数空间上的实例化
	
* 标记（label）：  

	>关于示例结果的信息称为“标记”（label）  
	
* 样例（sample）：  

	>拥有了标记信息的示例，称为“样例”（sample）  
	
* 输出空间：  

	>用 (x<sub>i</sub>,y<sub>i</sub>) 表示第 i 个样例，其中 y<sub>i</sub>∈Y 是示例 x<sub>i</sub> 的标记，
	<br>Y是所有标记的集合，亦称“标记空间”（label space）或“输出空间”  
	
* 分类（classification）：  

	>预测离散值的学习任务称为“分类”  
	
* 回归（regression）：  

	>预测连续值的学习任务称为“回归”  
	
* 二分类和多分类：  

	>对只涉及两个类别的“二分类”（binary classification）任务
	<br>通常称其中一种为“正类”（positive class）
	<br>另一类为“反类”（negative class）
	
	>涉及多个类别时，称为“多分类”（multi-class classification）任务  
	
* 预测任务：  
	
	>预测任务是希望通过对训练集 {(x1,y1),(x2,y2),...,(xm,ym)} 进行学习
	<br>建立一个从输入空间 X 到输出空间 Y 的映射 f:X→Y  

	>对二分类任务，通常令 Y={-1,+1} 或 {0,1}；
	
	>对多分类任务，
	$$
	|y|>2
	$$
	
	>对回归任务，$$Y=\mathbb{R}$$，$$\mathbb{R}$$为实数集  
	
* 测试：  

	>学得模型后，使用其进行预测的过程称为“测试”（testing）
	<br>被测试的样本称为“测试样本”（testing sample）
	<br>在学得f后，对测试例x，可得到其预测标记y=f(x)  
	
* 聚类：  

	>将训练集中的样本分成若干组，每组称为一个“簇”（cluster）
	<br>自动形成的簇可能对应一些潜在的概念，这些概念事先是不知道的
	<br>学习过程中使用的训练样本通常不拥有标记信息
	
* 监督和无监督：  

	>根据训练样本是否拥有标记信息
	<br>学习任务可大致划分为两类
	<br>“监督学习”（supervised learning）：如分类和回归
	<br>“无监督学习”（unsupervised learning）：如聚类
	
* 泛化能力（generalization）：  

	>学得模型适用于新样本的能力，称为“泛化”能力
	<br>具有强泛化能力的模型能够很好地适用于整个样本空间
	
* 独立同分布：  

	>通常假设样本空间中全体样本服从一个未知分布（distribution）D
	<br>我们获得的每个样本都是独立地从这个分布上采样获得的
	<br>即“独立同分布”（independent and identically distributed，简称i.i.d）
	
**1.3 假设空间**  

* 归纳学习（inductive learning）：

	>从特殊到一般的泛化过程称为归纳
	<br>“从样例中学习”是一个归纳的过程
	<br>因此称为“归纳学习”（inductive learning）
	
* 学习过程  

	>我们可以把学习过程看做一个在所有假设（hypothesis）组成的空间中进行搜索的过程
	<br>搜索的目标是找到与训练集“匹配”（fit）的假设
	<br>即能够将训练集中的样本判断正确的假设
	<br>假设的表示一旦确定，假设空间及其规模大小就确定了

* 学习结果  

	>可以有许多策略对假设空间进行搜索
	<br>搜索过程中可以不断删除与正例不一致的假设、与反例一致的假设
	<br>最终将会获得与训练集一致的假设
	<br>这就是我们学得的结果

* 版本空间（version space）：  

	>现实问题中常面临很大的假设空间
	<br>但学习过程基于有限样本训练集
	<br>因此可能有多个假设与训练集一致
	<br>即存在着一个与训练集一致的“假设集合”
	<br>我们称之为“版本空间”（version space）
	
**1.4 归纳偏好**  

* 归纳偏好（inductive bias）：  

	>机器学习在学习过程中对某类型假设的偏好，称为“归纳偏好”（inductive bias）
	<br>任何一个有效的机器学习算法必须有其归纳偏好
	<br>否则它将被假设空间中看似在训练集上“等效”的假设所迷惑
	<br>而无法产生确定的学习结果
	
* 奥卡姆剃刀（Occam's razor）：  

	>若有多个假设与观察一致，则选最简单的那个
	
* 没有免费的午餐（No Free Lunch Theorem）：  

	>任意两个算法的期望性能都相同，这就是NFL定理
	<br>NFL的前提是所有“问题”出现的机会相同，或所有问题同等重要
	<br>NFL假设真实目标函数f均匀分布，实际情况并非如此
	<br>学习算法自身的归纳偏好与问题是否相配，往往会起到决定性的作用
	
