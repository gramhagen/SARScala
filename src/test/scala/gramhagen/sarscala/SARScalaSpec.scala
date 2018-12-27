package gramhagen.sarscala

import org.apache.spark.api.java.JavaRDD
// import static org.apache.spark.ml.classification.LogisticRegressionSuite.generateLogisticInputAsList;
// import org.apache.spark.ml.feature.LabeledPoint;
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Row
import org.apache.spark.sql.RowFactory
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.DataTypes
import org.apache.spark.api.java.JavaSparkContext
import org.scalatest.fixture

class SARScalaSpec extends fixture.FlatSpec {

  case class FixtureParam(spark: SparkSession, jsc: JavaSparkContext, df: Dataset[Row])

  def withFixture(test: OneArgTest) = {
    var spark = SparkSession.builder()
        .master("local[2]") // 2 ... number of threads
        .appName(getClass().getSimpleName())
        // .config("spark.executor.memory", "50MB")
        // .config("spark.driver.memory", "50MB")
        .config("spark.sql.crossJoin.enabled", true)
        .config("spark.sql.shuffle.partitions", 1)
        .config("spark.ui.enabled", false)
        .getOrCreate()

    var jsc = new JavaSparkContext(spark.sparkContext)

    var col_user = Array(1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3)
    var col_item = Array(1, 2, 3, 4, 1, 2, 7, 8, 9, 10, 1, 2)
    var col_rating = Array(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)

    val range = 0 to col_user.length - 1

    var rowRDD = jsc.parallelize(range)
      .map(i => RowFactory.create(
            Int.box(col_user(i)),
            Int.box(col_item(i)),
            Int.box(col_rating(i))
          ))

    val schema = DataTypes.createStructType(Array(
      DataTypes.createStructField("user", DataTypes.IntegerType, false),
      DataTypes.createStructField("item", DataTypes.IntegerType, false), 
      DataTypes.createStructField("rating", DataTypes.IntegerType, false) 
    ))

    val df = spark.sqlContext.createDataFrame(rowRDD, schema).toDF()

    // TODO: remove me
    df.show()

    val theFixture = FixtureParam(spark, jsc, df)

    try {
      withFixture(test.toNoArgTest(theFixture))
    }
    finally {
      // TODO: still doesn't free all the memory... 
      rowRDD = null
      jsc.stop
      jsc.close
      jsc = null
      spark.stop
      spark.close
      spark = null
      SparkSession.clearDefaultSession();
      SparkSession.clearActiveSession();
    }
  }

  it should "be easy" in { f =>

    val sar = new SARScala()
    sar.setUserCol("user")
    sar.setItemCol("item")
    sar.setRatingCol("rating")

    val model = sar.fit(f.df)

    model.transform(f.df).show()

    model.itemSimilarity.show()
  }
}