package gramhagen.sarscala

import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.apache.spark.api.java.JavaSparkContext
import org.scalatest.{Outcome, fixture}
import org.scalatest.Matchers._


class SARScalaSpec extends fixture.FlatSpec {

  case class FixtureParam(spark: SparkSession, jsc: JavaSparkContext, df: Dataset[Row], coOccurrence: Dataset[Row], similarity: Dataset[Row])

  def withFixture(test: OneArgTest): Outcome = {
    var spark = SparkSession.builder()
        .master("local[2]") // 2 ... number of threads
        .appName("SARScalaSpec")
        .config("spark.sql.shuffle.partitions", value = 1)
        .config("spark.ui.enabled", value = false)
        .getOrCreate()

    var jsc = new JavaSparkContext(spark.sparkContext)

    var rows = jsc.parallelize(Seq(
      Row(1, 1, 1), Row(1, 2, 1), Row(1, 3, 1), Row(1, 4, 1),
      Row(2, 1, 1), Row(2, 2, 1), Row(2, 7, 1), Row(2, 8, 1), Row(2, 9, 1), Row(2, 10, 1),
      Row(3, 1, 1), Row(3, 2, 1)))

    var schema = StructType(Array(
      StructField("user", IntegerType, nullable = false),
      StructField("item", IntegerType, nullable = false),
      StructField("rating", IntegerType, nullable = false)
    ))

    val df = spark.sqlContext.createDataFrame(rows, schema)

    rows = jsc.parallelize(Seq(
      Row(1, 1, 3), Row(1, 2, 3), Row(1, 3, 1), Row(1, 4, 1), Row(1, 7, 1), Row(1, 8, 1), Row(1, 9, 1), Row(1, 10, 1),
      Row(2, 2, 3), Row(2, 3, 1), Row(2, 4, 1), Row(2, 7, 1), Row(2, 8, 1), Row(2, 9, 1), Row(2, 10, 1),
      Row(3, 3, 1), Row(3, 4, 1),
      Row(4, 4, 1),
      Row(7, 7, 1), Row(7, 8, 1), Row(7, 9, 1), Row(7, 10, 1),
      Row(8, 8, 1), Row(8, 9, 1), Row(8, 10, 1),
      Row(9, 9, 1), Row(9, 10, 1),
      Row(10, 10, 1)))

    schema = StructType(Array(
      StructField("i1", IntegerType, nullable = false),
      StructField("i2", IntegerType, nullable = false),
      StructField("count", IntegerType, nullable = false)
    ))

    val coOccurrence = spark.sqlContext.createDataFrame(rows, schema)

    rows = jsc.parallelize(Seq(
      Row(1, 1, 1.0), Row(1, 2, 1.0), Row(1, 3, 0.333), Row(1, 4, 0.333), Row(1, 7, 0.333), Row(1, 8, 0.333), Row(1, 9, 0.333), Row(1, 10, 0.333),
      Row(2, 1, 1.0), Row(2, 2, 1.0), Row(2, 3, 0.333), Row(2, 4, 0.333), Row(2, 7, 0.333), Row(2, 8, 0.333), Row(2, 9, 0.333), Row(2, 10, 0.333),
      Row(3, 1, 0.333), Row(3, 2, 0.333), Row(3, 3, 1.0), Row(3, 4, 1.0),
      Row(4, 1, 0.333), Row(4, 2, 0.333), Row(4, 3, 1.0), Row(4, 4, 1.0),
      Row(7, 1, 0.333), Row(7, 2, 0.333), Row(7, 7, 1.0), Row(7, 8, 1.0), Row(7, 9, 1.0), Row(7, 10, 1.0),
      Row(8, 1, 0.333), Row(8, 2, 0.333), Row(8, 7, 1.0), Row(8, 8, 1.0), Row(8, 9, 1.0), Row(8, 10, 1.0),
      Row(9, 1, 0.333), Row(9, 2, 0.333), Row(9, 7, 1.0), Row(9, 8, 1.0), Row(9, 9, 1.0), Row(9, 10, 1.0),
      Row(10, 1, 0.333), Row(10, 2, 0.333), Row(10, 7, 1.0), Row(10, 8, 1.0), Row(10, 9, 1.0), Row(10, 10, 1.0)))

    schema = StructType(Array(
      StructField("i1", IntegerType, nullable = false),
      StructField("i2", IntegerType, nullable = false),
      StructField("value", DoubleType, nullable = false)
    ))

    val similarity = spark.sqlContext.createDataFrame(rows, schema)

    val theFixture = FixtureParam(spark, jsc, df, coOccurrence, similarity)

    try {
      withFixture(test.toNoArgTest(theFixture))
    }
    finally {
      // TODO: still doesn't free all the memory... 
      rows = null
      jsc.stop
      jsc.close
      jsc = null
      spark.stop
      spark.close
      spark = null
      SparkSession.clearDefaultSession()
      SparkSession.clearActiveSession()
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

  it should "calculate item co-occurrence" in { f =>

    val expectedI1 = Array(1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 4, 7, 7, 7, 7, 8, 8, 8, 9, 9, 10)
    val expectedI2 = Array(1, 2, 3, 4, 7, 8, 9, 10, 2, 3, 4, 7, 8, 9, 10, 3, 4, 4, 7, 8, 9, 10, 8, 9, 10, 9, 10, 10)
    val expectedCount = Array(3, 3, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)

    val actual = new SARScala()
      .setUserCol("user")
      .setItemCol("item")
      .getItemCoOcurrence(f.df)
      .orderBy("i1", "i2")
      .toDF()
      .rdd

    assert(actual.map(r => r(0)).collect() sameElements expectedI1)
    assert(actual.map(r => r(1)).collect() sameElements expectedI2)
    assert(actual.map(r => r(2)).collect() sameElements expectedCount)
  }

  it should "calculate item-item similarity" in { f =>

    val expectedI1 = Array(
      1, 1, 1, 1, 1, 1, 1, 1,
      2, 2, 2, 2, 2, 2, 2, 2,
      3, 3, 3, 3,
      4, 4, 4, 4,
      7, 7, 7, 7, 7, 7,
      8, 8, 8, 8, 8, 8,
      9, 9, 9, 9, 9, 9,
      10, 10, 10, 10, 10, 10)

    val expectedI2 = Array(
      1, 2, 3, 4, 7, 8, 9, 10,
      1, 2, 3, 4, 7, 8, 9, 10,
      1, 2, 3, 4,
      1, 2, 3, 4,
      1, 2, 7, 8, 9, 10,
      1, 2, 7, 8, 9, 10,
      1, 2, 7, 8, 9, 10,
      1, 2, 7, 8, 9, 10)

    val expectedValues = Array(
      1, 1, 0.333, 0.333, 0.333, 0.333, 0.333, 0.333,
      1, 1, 0.333, 0.333, 0.333, 0.333, 0.333, 0.333,
      0.333, 0.333, 1, 1,
      0.333, 0.333, 1, 1,
      0.333, 0.333, 1, 1, 1, 1,
      0.333, 0.333, 1, 1, 1, 1,
      0.333, 0.333, 1, 1, 1, 1,
      0.333, 0.333, 1, 1, 1, 1)

    val actual = new SARScala()
      .setUserCol("user")
      .setItemCol("item")
      .getItemSimilarity(f.coOccurrence, "jaccard")
      .orderBy("i1", "i2")
      .toDF()
      .rdd

    assert(actual.map(r => r(0)).collect() sameElements expectedI1)
    assert(actual.map(r => r(1)).collect() sameElements expectedI2)

    val actualValues = actual.map(r => r(2).asInstanceOf[Double]).collect()
    assert(actualValues.length === expectedValues.length)

    for (i <- actualValues.indices) {
      actualValues(i) should equal(expectedValues(i) +- 0.001)
    }
  }

  it should "get mapped arrays" in { f =>
    val model = new SARScalaModel("uid_1", f.similarity)
    val (counts, indices, values) = model.getMappedArrays

    assert (counts === Array(8, 8, 4, 4, 6, 6, 6, 6))

    assert (indices === Array(
      0, 1, 2, 3, 4, 5, 6, 7,
      0, 1, 2, 3, 4, 5, 6, 7,
      0, 1, 2, 3,
      0, 1, 2, 3,
      0, 1, 4, 5, 6, 7,
      0, 1, 4, 5, 6, 7,
      0, 1, 4, 5, 6, 7,
      0, 1, 4, 5, 6, 7
    ))

    assert (values === Array(
      1.0, 1.0, 0.333, 0.333, 0.333, 0.333, 0.333, 0.333,
      1.0, 1.0, 0.333, 0.333, 0.333, 0.333, 0.333, 0.333,
      0.333, 0.333, 1.0, 1.0,
      0.333, 0.333, 1.0, 1.0,
      0.333, 0.333, 1.0, 1.0, 1.0, 1.0,
      0.333, 0.333, 1.0, 1.0, 1.0, 1.0,
      0.333, 0.333, 1.0, 1.0, 1.0, 1.0,
      0.333, 0.333, 1.0, 1.0, 1.0, 1.0))

  }

}