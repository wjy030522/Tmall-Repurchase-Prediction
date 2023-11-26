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

各数据集的具体字段信息可前往[天猫复购数据集](https://www.heywhale.com/mw/dataset/622d9ebf8a84f900178990ec)

## 三、对数据集的分析与探索

### 分析不同年龄层，不同群体的用户购买倾向（利用pandas透视表快速地进行分类汇总）

![image](https://github.com/wjy030522/Tmall-Repurchase-Prediction/assets/108457536/0cfd9329-f1dd-4933-8243-e765592e198a)
浏览人数呈上升趋势，于11月达到峰值18337352，推测为受到双11活动的影响而购买率则是在5月为峰值10.347%，再来是11月7.362%，收藏、加购、购买三个行为的趋势变化一致，但是购买和收藏次数基本一致，加入购物车的次数却明显少于二者。
