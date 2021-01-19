from __future__ import print_function

import re
import sys

from operator import add

from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint

import matplotlib.pyplot as plt

if __name__ == "__main__":
    sc = SparkContext(appName="final_project")

    # read data
    data_with_header = sc.textFile('Jan_2019_ontime.csv').map(lambda x: x.split(","))
    header = sc.parallelize([data_with_header.first()])
    data = data_with_header.zipWithIndex().filter(lambda x: x[1] > 0).map(lambda x: x[0]) 
    data.cache()

    row = data.count()
    print(row)

    col = data.map(lambda x:len(x)).collect()[0]
    print(col)

    # label distribution
    positive = data.filter(lambda x: x[17]=='1')
    negative = data.filter(lambda x: x[17]=='0')
    positive.cache()
    negative.cache()
    labels=['delay','on time']
    X=[positive.count(),negative.count()] 
    fig = plt.figure()
    plt.pie(X,labels=labels,autopct='%1.2f%%')
    plt.title("flights arrive type")

    # attributes selection
    def visual(positive,negative,plot_type,att_index,att_type):
        if att_type == 'numeric':
            att_delay = positive.map(lambda x: (int(x[att_index]),1)).reduceByKey(add)
            att_ontime = negative.map(lambda x: (int(x[att_index]),1)).reduceByKey(add)
        else:
            att_delay = positive.map(lambda x: (x[att_index],1)).reduceByKey(add)
            att_ontime = negative.map(lambda x: (x[att_index],1)).reduceByKey(add)        
        att_ontime.cache()
        att_delay.cache()
        att_percentage_delay = att_ontime.join(att_delay).map(lambda x: (x[0],x[1][1]/(x[1][0]+x[1][1]))).sortByKey()
        labels = att_percentage_delay.flatMap(lambda x: (x[0],)).collect()
        percentage = att_percentage_delay.flatMap(lambda x: (x[1],)).collect()
        fig=plt.figure()
        if plot_type == 'line':
            plt.plot(labels,percentage)
        elif plot_type == 'bar':
            ind = [x for x,_ in enumerate(labels)]
            plt.bar(ind, percentage)
            plt.xticks(ind, labels)
            fig.autofmt_xdate() 
        else:
            plt.scatter(labels,percentage,s=2)
            plt.xticks([])
        plt.show()

    # 0 'DAY_OF_MONTH': Day of the month
    visual(positive,negative,'line',0,'numeric')

    # 1 'DAY_OF_WEEK': Day of the week
    visual(positive,negative,'line',1,'numeric')

    # 2 'OP_UNIQUE_CARRIER': Unique tranport code & 3 OP_CARRIER_AIRLINE_ID: Unique aviation operator code & 4 'OP_CARRIER': IATA code of the operator
    visual(positive,negative,'bar',2,'string')

    # 5 'TAIL_NUM': Tail number
    visual(positive,negative,'scatter',5,'string')
    # decrease the levels of attributes
    tail_number = data.map(lambda x: (x[5],1)).reduceByKey(add).sortBy(lambda x: x[1],False)
    print(tail_number.count())
    tailnumber_delay = positive.map(lambda x: (x[5],1)).reduceByKey(add)
    tailnumber_delay_ontime = negative.map(lambda x: (x[5],1)).reduceByKey(add)        
    tailnumber_percentage_delay_count = tailnumber_delay_ontime.join(tailnumber_delay).map(lambda x: (x[0],x[1][1]/(x[1][0]+x[1][1]))).join(tail_number)
    data_5 = data.map(lambda x: (x[5],(x[0],x[1],x[2],x[3],x[4],x[6],x[7],x[8],x[9],x[10],x[11],x[12],x[13],x[14],x[15],x[16],x[17],x[18],x[19],x[20])))
    print(tailnumber_percentage_delay_count.take(5))
    data_5_delayrate_count = data_5.join(tailnumber_percentage_delay_count)
    data_new_5 = data_5_delayrate_count.map(lambda x: (x[1][0],x[0] if (x[1][1][0] > 0.2 or x[1][1][0] < 0.05) and x[1][1][1]>150 else 'others'))
    data = data_new_5.map(lambda x: (x[0][0],x[0][1],x[0][2],x[0][3],x[0][4],x[1],x[0][5],x[0][6],x[0][7],x[0][8],x[0][9],x[0][10],x[0][11],x[0][12],x[0][13],x[0][14],x[0][15],x[0][16],x[0][17],x[0][18],x[0][19]))
    print(data.map(lambda x:x[5]).distinct().count())

    # 6 'OP_CARRIER_FL_NUM': Flight number
    visual(positive,negative,'scatter',6,'string')

    flight_number = data.map(lambda x: (x[6],1)).reduceByKey(add).sortBy(lambda x: x[1],False)
    print(flight_number.count())
    flightnumber_delay = positive.map(lambda x: (x[6],1)).reduceByKey(add)
    flightnumber_delay_ontime = negative.map(lambda x: (x[6],1)).reduceByKey(add)        
    flightnumber_percentage_delay_count = flightnumber_delay_ontime.join(flightnumber_delay).map(lambda x: (x[0],x[1][1]/(x[1][0]+x[1][1]))).join(flight_number)
    data_6 = data.map(lambda x: (x[6],(x[0],x[1],x[2],x[3],x[4],x[5],x[7],x[8],x[9],x[10],x[11],x[12],x[13],x[14],x[15],x[16],x[17],x[18],x[19],x[20])))
    data_6_delayrate_count = data_6.join(flightnumber_percentage_delay_count)
    data_new_6 = data_6_delayrate_count.map(lambda x: (x[1][0],x[0] if (x[1][1][0] > 0.2 or x[1][1][0] < 0.05) and x[1][1][1]>150 else 'others'))
    data = data_new_6.map(lambda x: (x[0][0],x[0][1],x[0][2],x[0][3],x[0][4],x[0][5],x[1],x[0][6],x[0][7],x[0][8],x[0][9],x[0][10],x[0][11],x[0][12],x[0][13],x[0][14],x[0][15],x[0][16],x[0][17],x[0][18],x[0][19]))
    print(data.map(lambda x:x[6]).distinct().count())

    # 7 'ORIGIN_AIRPORT_ID': Origin airport ID & 8 'ORIGIN_AIRPORT_SEQ_ID': Origin airport ID - SEQ & 9 'ORIGIN': Airport of Origin
    visual(positive,negative,'scatter',9,'string')
    origin = data.map(lambda x: (x[9],1)).reduceByKey(add).sortBy(lambda x: x[1],False)
    print(origin.count())
    origin_delay = positive.map(lambda x: (x[9],1)).reduceByKey(add)
    origin_delay_ontime = negative.map(lambda x: (x[9],1)).reduceByKey(add)        
    origin_percentage_delay_count = origin_delay_ontime.join(origin_delay).map(lambda x: (x[0],x[1][1]/(x[1][0]+x[1][1]))).join(origin)
    data_9 = data.map(lambda x: (x[9],(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[10],x[11],x[12],x[13],x[14],x[15],x[16],x[17],x[18],x[19],x[20])))
    data_9_delayrate_count = data_9.join(origin_percentage_delay_count)
    data_new_9 = data_9_delayrate_count.map(lambda x: (x[1][0],x[0] if x[1][1][0] > 0.2 or x[1][1][0] < 0.1 else 'others'))
    data = data_new_9.map(lambda x: (x[0][0],x[0][1],x[0][2],x[0][3],x[0][4],x[0][5],x[0][6],x[0][7],x[0][8],x[1],x[0][9],x[0][10],x[0][11],x[0][12],x[0][13],x[0][14],x[0][15],x[0][16],x[0][17],x[0][18],x[0][19]))
    print(data.map(lambda x:x[9]).distinct().count())

    # 10 'DEST_AIRPORT_ID': ID of the destination airport & 11 'DEST_AIRPORT_SEQ_ID': Destination airport ID - SEQ & 12 'DEST': Destination airport
    visual(positive,negative,'scatter',12,'string')
    print(data.map(lambda x:x[12]).distinct().count())
    dest = data.map(lambda x: (x[12],1)).reduceByKey(add).sortBy(lambda x: x[1],False)
    dest_delay = positive.map(lambda x: (x[12],1)).reduceByKey(add)
    dest_delay_ontime = negative.map(lambda x: (x[12],1)).reduceByKey(add)        
    dest_percentage_delay_count = dest_delay_ontime.join(dest_delay).map(lambda x: (x[0],x[1][1]/(x[1][0]+x[1][1]))).join(dest)
    data_12 = data.map(lambda x: (x[12],(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9],x[10],x[11],x[13],x[14],x[15],x[16],x[17],x[18],x[19],x[20])))
    data_12_delayrate_count = data_12.join(dest_percentage_delay_count)
    data_new_12 = data_12_delayrate_count.map(lambda x: (x[1][0],x[0] if x[1][1][0] > 0.2 or x[1][1][0] < 0.1 else 'others'))
    data = data_new_12.map(lambda x: (x[0][0],x[0][1],x[0][2],x[0][3],x[0][4],x[0][5],x[0][6],x[0][7],x[0][8],x[0][9],x[0][10],x[0][11],x[1],x[0][12],x[0][13],x[0][14],x[0][15],x[0][16],x[0][17],x[0][18],x[0][19]))
    print(data.map(lambda x:x[12]).distinct().count())

    # 13 'DEP_TIME': Flight departure time & 15 'DEP_TIME_BLK': block of time (hour) where the match has been postponed
    visual(positive,negative,'line',15,'string')

    # 14 'DEP_DEL15': Departure delay indicator
    visual(positive,negative,'bar',14,'string')

    # 16 'ARR_TIME'
    visual(positive,negative,'scatter',16,'int')

    def arr_time(x):
        if x >= 600 and x <= 659:
            return '0600-0659'
        elif x>=1400 and x<=1459:
            return '1400-1459'
        elif x>=1200 and x<=1259:
            return '1200-1259'
        elif x>=1500 and x<=1559:
            return '1500-1559'
        elif x>=1900 and x<=1959:
            return '1900-1959'
        elif x>=900 and x<=959:
            return '0900-0959'
        elif x>=1000 and x<=1059:
            return  '1000-1059'
        elif x>=2000 and x<=2059:
            return '2000-2059'
        elif x>=1300 and x<=1359:
            return '1300-1359'
        elif x>=1100 and x<=1159:
            return '1100-1159'
        elif x>=800 and x<=859:
            return '0800-0859'
        elif x>=2200 and x<=2259:
            return '2200-2259'
        elif x>=1600 and x<=1659:
            return '1600-1659'
        elif x>=1700 and x<=1759:
            return '1700-1759'
        elif x>=2100 and x<=2159:
            return '2100-2159'
        elif x>=700 and x<=759:
            return '0700-0759'
        elif x>=1800 and x<=1859:
            return '1800-1859'
        elif x>=1 and x<=559:
            return '0001-0559'
        elif x>=2300 and x<=2400:
            return '2300-2400'

    data1 = data.filter(lambda x:x[16]!='').map(lambda x:(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9],x[10],x[11],x[12],x[13],x[14],x[15],arr_time(int(x[16])),x[17],x[18],x[19],x[20]))

    print(data1.take(10))

    # 19 'DIVERTED': Indicator if the flight has been diverted
    visual(positive,negative,'bar',19,'string')

    # 20 'DISTANCE': Distance between airports
    visual(positive,negative,'scatter',20,'int')

    # data cleaning
    cleaned_data = data1.filter(lambda x: x[17]!='' and x[0]!='' and x[1]!='' and x[2]!='' and x[3]!='' and x[5]!='' and x[6]!='' and x[9]!='' and x[12]!='' and x[14] !='' and x[15]!='' and x[20]!='' and x[18]=='0')
    print(cleaned_data.take(10))

    col_name = ['label','day_of_month','day_of_week','tranport_code','tail_number','flight_number','origin_airport','destination_airtport','departure_delay','departure_time_block','arrive_time_block','distance']
    numericCols = ['day_of_month', 'day_of_week', 'distance']
    categoricalColumns = ['tranport_code','tail_number','flight_number','origin_airport','destination_airtport','departure_delay','departure_time_block','arrive_time_block']
    labels_features = cleaned_data.map(lambda x: (int(x[17]),int(x[0]),int(x[1]),x[2],x[5],x[6],x[9],x[12],x[14],x[15],x[16],int(x[20])))

    # model building
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import StringIndexer
    from pyspark.ml.classification import RandomForestClassifier
    from pyspark.ml.feature import VectorAssembler
    from pyspark.sql.functions import split, explode, concat, concat_ws
    import numpy as np
    from pyspark.sql.functions import udf
    from pyspark.sql.types import *
    import pyspark.sql.functions as f
    from pyspark.ml.linalg import Vectors, VectorUDT

    data_df1 = labels_features.toDF(col_name)
    # sampling
    data_df = data_df1.stat.sampleBy("label", {0:0.25,1:1}, 7)
    # spliting
    train,test = data_df.randomSplit([0.7, 0.3], 7)

    print(train.show())

    def data_format(data):
        
        indexers = [StringIndexer(inputCol=col, outputCol=col+"_index").fit(data) for col in categoricalColumns]
        pipeline = Pipeline(stages=indexers)
        data_features = pipeline.fit(data).transform(data)

        features_withlabel = ['label']+[c+"_index" for c in categoricalColumns] + numericCols
        data_split = data_features.select(features_withlabel)
        features = [f.col(c+"_index") for c in categoricalColumns] + [f.col(col) for col in numericCols]
        data_label_features = data_split.withColumn("features",f.array(features)).select('label','features')
        
        list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())
        df_with_vectors = data_label_features.select(
            data_label_features["label"], 
            list_to_vector_udf(data_label_features["features"]).alias('features')
        )
        return df_with_vectors

    train_df = data_format(train)
    train_df.cache()

    print(train_df.show(3,False))

    rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=300)
    rfModel = rf.fit(train_df)

    importances = []
    for i in range(len(col_name)-1):
        importances.append(rfModel.featureImportances[i])
    x = categoricalColumns+numericCols

    plt.figure()
    plt.title("Feature importances")
    plt.xticks(rotation=90)
    plt.bar(x, importances)
    plt.show

    # prediction
    test_df = data_format(test)
    predictResult = rfModel.transform(test_df)

    # evaluation
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator

    TP = predictResult.filter((predictResult.label==1)&(predictResult.prediction==1)).count()
    FN = predictResult.filter((predictResult.label==1)&(predictResult.prediction==0)).count()
    TN = predictResult.filter((predictResult.label==0)&(predictResult.prediction==0)).count()
    FP = predictResult.filter((predictResult.label==0)&(predictResult.prediction==1)).count()

    F1 = 2*TP/(2*TP+FP+FN)

    print(TP)
    print(FN)
    print(TN)
    print(FP)
    print(F1)

    recall = TP/(TP+FN)
    precision = TP/(TP+FP)

    print(recall)
    print(precision)

    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictResult)
    print(accuracy)

    sc.stop()



