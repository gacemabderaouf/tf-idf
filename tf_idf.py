
# /usr/local/spark/bin/spark-submit tf_idf.py 
from pyspark import SparkContext
from pyspark import rdd
import os
import datetime
import math
import time

if os.path.exists("output.txt"):
  os.remove("output.txt")

f= open("output.txt","w+")
sc = SparkContext("local", "tf_idf")

tf_rdd = sc.parallelize([])
idf_rdd = sc.parallelize([])
#files = os.listdir("/home/selma/Desktop/idf/files/")
files = ["file{0}.txt".format(x) for x in range(1,6)]
start_time = time.time()
for input_file in files:
  data = sc.textFile("file:///home/selma/Desktop/idf/files/"+input_file)
  words = data.flatMap(lambda x: x.split(' ')) 
  
  tf_counts = words.map(lambda x: (x, 1))
  idf_counts = tf_counts.reduceByKey(lambda a, b: 1)

  tf_rdd = tf_rdd.union(tf_counts)
  idf_rdd = idf_rdd.union(idf_counts)


tf_count = tf_rdd.reduceByKey(lambda a, b: a+b).collect()
idf_count = idf_rdd.reduceByKey(lambda a, b: a+b).collect()
all_words = words.count()
all_files = len(files)
exec_time = time.time() - start_time

f.write(str(datetime.datetime.now())+"\n + Exec time : "+str(exec_time)+"\n + number of files : "+str( all_files)+"\n")
f.write("    tf            idf\n")

i=0
for (word, count) in tf_count:
  tf = count/float(all_words)
  idf = math.log(all_files/float(idf_count[i][1]))
  f.write("%s: %f      %f\n" % (word, tf, idf))
  i+=1

f.close()


