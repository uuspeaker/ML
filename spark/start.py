from pyspark import SparkConf, SparkContext
conf = SparkConf().setMaster("local").setAppName("My App")
sc = SparkContext(conf = conf)

lines = sc.textFile("dataset/README.md")

def hasPython(line):
    return "Python" in line

pythonLines = lines.filter(hasPython)

print("pythonLines",pythonLines)