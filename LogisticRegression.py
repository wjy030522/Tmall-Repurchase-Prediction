from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, ChiSqSelector, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
conf = SparkConf().setAppName("lghg").setMaster("spark://master:7077")
sc = SparkContext(conf=conf)
sc.setLogLevel("WARN")   # 设置日志级别
spark = SparkSession(sc)

print("load spark successful")
# 读取训练集和测试集的数据
train = spark.read.csv('hdfs://master:9000/project/train.csv', header=True, inferSchema=True)
test = spark.read.csv('hdfs://master:9000/project/test.csv', header=True, inferSchema=True)

# 从训练集中提取特征(X)和标签(Y)
feature_columns = train.columns[3:]
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
train = assembler.transform(train)
test = assembler.transform(test)

# 使用卡方检验选择特征
selector = ChiSqSelector(numTopFeatures=5, featuresCol="features", outputCol="selectedFeatures", labelCol="label")
model = selector.fit(train)
train = model.transform(train)
test = model.transform(test)

# 标准化处理
scaler = StandardScaler(inputCol="selectedFeatures", outputCol="scaledFeatures", withMean=True, withStd=True)
scalerModel = scaler.fit(train)
train = scalerModel.transform(train)
test = scalerModel.transform(test)

# 从训练集中提取特征(X)和标签(Y)
data = train.select("label", "scaledFeatures")
# 划分训练集和测试集
(trainingData, testData) = data.randomSplit([0.7, 0.3], seed=1)

# 使用逻辑回归模型进行训练和评估
lr = LogisticRegression(featuresCol="scaledFeatures", labelCol="label", maxIter=10, regParam=0.01)
model = lr.fit(trainingData)
predictions = model.transform(testData)
# 使用BinaryClassificationEvaluator计算准确率
evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
accuracy = evaluator.evaluate(predictions)
print("准确率:", accuracy)


# 创建MulticlassClassificationEvaluator
evaluator_multiclass = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

# 计算精确率、召回率和F1分数
precision = evaluator_multiclass.evaluate(predictions, {evaluator_multiclass.metricName: "weightedPrecision"})
recall = evaluator_multiclass.evaluate(predictions, {evaluator_multiclass.metricName: "weightedRecall"})
f1_score = evaluator_multiclass.evaluate(predictions, {evaluator_multiclass.metricName: "f1"})

# 打印结果
print("精确率:", precision)
print("召回率:", recall)
print("F1分数:", f1_score)


