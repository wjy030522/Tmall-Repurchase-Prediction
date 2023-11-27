# 天猫复购预测
## 一、项目背景

本项目基于阿里天池大数据竞赛的天猫复购预测学习赛，基于赛题，我们首先对数据进行分析和可视化并得出结论，然后从数据集中提取可能对预测标签有影响的特征形成新的训练集，然后搭建了Hadoop+spark集群进行算法设计实现，通过运用预测模型算法，分析挖掘所提供的数据，预测消费者会复购，成为忠实、粘性客户的可能。

## 二、数据集说明  

数据集包含了匿名用户在 "双十一 "前6个月和"双十一 "当天的购物记录，标签为是否是重复购买者。出于隐私保护，数据采样存在部分偏差，该数据集的统计结果会与天猫的实际情况有一定的偏差，但不影响解决方案的适用性。本次学习赛提供了两种格式的数据集，为了更方便做特征工程，我们选择了data_format1.zip文件夹（内含4个文件），数据描述如下。

数据集包含以下四个文件：  
```
用户行为日志 user_log_format1.csv  
用户画像 user_info_format1.csv  
训练数据 train_format1.csv  
测试数据 test_format1.csv  
```

用户行为日志未放到github上，用户行为数据集数据集和各数据集的具体字段信息可前往[天猫复购数据集](https://www.heywhale.com/mw/dataset/622d9ebf8a84f900178990ec)

## 三、对数据集的分析与探索

### 分析不同年龄层，不同群体的用户购买倾向（利用pandas透视表快速地进行分类汇总）

![image](https://github.com/wjy030522/Tmall-Repurchase-Prediction/assets/108457536/0cfd9329-f1dd-4933-8243-e765592e198a)
浏览人数呈上升趋势，于11月达到峰值18337352，推测为受到双11活动的影响，而购买率则是在5月为峰值10.347%，再来是11月7.362%，收藏、加购、购买三个行为的趋势变化一致，但是购买和收藏次数基本一致，加入购物车的次数却明显少于二者。

![image](https://github.com/wjy030522/Tmall-Repurchase-Prediction/assets/108457536/767b3cde-d23c-47e7-be0a-087fcef5efbf)
其中11号的购买率最高，推测是双11影响

![image](https://github.com/wjy030522/Tmall-Repurchase-Prediction/assets/108457536/508fed89-dab9-4dc6-8c4b-cfe02c48df22)
对11月份数据剔除后进行分析，发现月中的购买率比较高，月底的购买率逐渐下降，而浏览、加购、收藏次数则是从月初到月底逐渐增多。

![image](https://github.com/wjy030522/Tmall-Repurchase-Prediction/assets/108457536/6a9bbf4f-4fb8-4e54-a584-45886081d0d1)
<18岁为1；[18,24]为2；[25,29]为3； [30,34]为4；[35,39]为5；[40,49]为6； > = 50时为7和8; 0和NULL表示未知，可以发现 小于18岁的客户非常少，[25,29]浏览最多，而购买率最高的是[30,34]的客户。

## 四、数据的预处理

用pandas透视表(pivot_table)从用户行为日志中提取商家特征，用户特征，用户在特定商家下的特征以及用户画像中的年龄、性别等字段，形成新的训练集。最终形成260864个含有25个字段的样本。

## 五、算法的设计与实现（hadoop+spark集群下运行）

### 卡方检验选择特征

使用Spark MLlib库中的一个特征选择器ChiSqSelector，它基于卡方检验进行特征选择。此次项目选择卡方值最高的5个特征。
```
# 使用卡方检验选择特征
selector = ChiSqSelector(numTopFeatures=5, featuresCol="features", outputCol="selectedFeatures", labelCol="label")
model = selector.fit(train)
train = model.transform(train)
test = model.transform(test)
```

### 数据标准化处理

使用Spark MLlib库中的一个特征变换器StandardScaler，用于标准化特征。将前面卡方检验选择的特征列进行标准化处理。
```
# 标准化处理
scaler = StandardScaler(inputCol="selectedFeatures", outputCol="scaledFeatures", withMean=True, withStd=True)
scalerModel = scaler.fit(train)
train = scalerModel.transform(train)
test = scalerModel.transform(test)
```

### 模型训练与评估

### 逻辑回归
```
# 使用逻辑回归模型进行训练和评估
lr = LogisticRegression(featuresCol="scaledFeatures", labelCol="label", maxIter=10, regParam=0.01)
model = lr.fit(trainingData)
predictions = model.transform(testData)
```
### 评估结果
```
加权精确率：0.8959
加权召回率：0.9388
F1-score：0.9095
```

### 随机森林
```
# 创建随机森林模型
rf = RandomForestClassifier(featuresCol="scaledFeatures", labelCol="label", numTrees=10, maxDepth=5, seed=1)

# 在训练集上拟合模型
rf_model = rf.fit(trainingData)

# 在测试集上进行预测
predictions = rf_model.transform(testData)
```
### 评估结果
```
加权精确率：0.8817
加权召回率：0.9390
F1-score：0.9094
```
