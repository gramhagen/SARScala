package gramhagen.sarscala

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.scalatest.{Outcome, fixture}


class SARScalaSpec extends fixture.FlatSpec {

  case class FixtureParam(data: Map[String, DataFrame])

  def withFixture(test: OneArgTest): Outcome = {
    var spark = SparkSession.builder()
      .master("local[2]") // 2 ... number of threads
      .appName("SARScalaSpec")
      .config("spark.sql.shuffle.partitions", value = 1)
      .config("spark.ui.enabled", value = false)
      .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    // load dataframes from csv files in resources directory
    val dataFiles = Seq(
      "getProcessedRatingsInput",
      "getItemCoOccurrenceInput",
      "getItemSimilarityInput",
      "getUserAffinityInput",
      "itemSimilarity",
      "processedRatings")

    val data = dataFiles.map((name: String) => {
      (name, spark.sqlContext
        .read
        .format("csv")
        .option("header", "true")
        .option("inferSchema", "true")
        .load(getClass.getResource(f"/$name%s.csv").getPath))
    }).toMap

    val theFixture = FixtureParam(data)

    try {
      withFixture(test.toNoArgTest(theFixture))
    }
    finally {
      spark.stop
      spark = null
      SparkSession.clearDefaultSession()
      SparkSession.clearActiveSession()
    }
  }

  it should "build and fit" in { f =>

    val sar = new SARScala()
      .setUserCol("user")
      .setItemCol("item")
      .setRatingCol("rating")
      .setTimeCol("time")
    assert(sar.isInstanceOf[SARScala])

    val model = sar.fit(f.data.apply("getProcessedRatingsInput"))
    assert(model.isInstanceOf[SARScalaModel])
  }

  it should "calculate item co-occurrence" in { f =>

    val expected = Seq(
      (1, 1, 3), (1, 2, 3), (1, 3, 2), (1, 4, 1), (1, 7, 1), (1, 8, 1), (1, 9, 1), (1, 10, 1),
      (2, 2, 3), (2, 3, 2), (2, 4, 1), (2, 7, 1), (2, 8, 1), (2, 9, 1), (2, 10, 1),
      (3, 3, 2), (3, 4, 1),
      (4, 4, 1),
      (7, 7, 1), (7, 8, 1), (7, 9, 1), (7, 10, 1),
      (8, 8, 1), (8, 9, 1), (8, 10, 1),
      (9, 9, 1), (9, 10, 1),
      (10, 10, 1)
    )

    new SARScala()
      .setUserCol("user")
      .setItemCol("item")
      .getItemCoOccurrence(f.data.apply("getItemCoOccurrenceInput"))
      .orderBy("i1", "i2")
      .toDF()
      .collect()
      .zip(expected)
      .foreach({ case (row, testValue) =>
        assert(row.getAs[Int]("i1") === testValue._1)
        assert(row.getAs[Int]("i2") === testValue._2)
        assert(row.getAs[Int]("count") === testValue._3)
      })
  }

  it should "calculate item-item similarity - jaccard" in { f =>
    val expected = Seq(
      (1, 1, 1.0), (1, 2, 1.0), (1, 3, 0.6), (1, 4, 0.3), (1, 7, 0.3), (1, 8, 0.3), (1, 9, 0.3), (1, 10, 0.3),
      (2, 1, 1.0), (2, 2, 1.0), (2, 3, 0.6), (2, 4, 0.3), (2, 7, 0.3), (2, 8, 0.3), (2, 9, 0.3), (2, 10, 0.3),
      (3, 1, 0.6), (3, 2, 0.6), (3, 3, 1.0), (3, 4, 0.5),
      (4, 1, 0.3), (4, 2, 0.3), (4, 3, 0.5), (4, 4, 1.0),
      (7, 1, 0.3), (7, 2, 0.3), (7, 7, 1.0), (7, 8, 1.0), (7, 9, 1.0), (7, 10, 1.0),
      (8, 1, 0.3), (8, 2, 0.3), (8, 7, 1.0), (8, 8, 1.0), (8, 9, 1.0), (8, 10, 1.0),
      (9, 1, 0.3), (9, 2, 0.3), (9, 7, 1.0), (9, 8, 1.0), (9, 9, 1.0), (9, 10, 1.0),
      (10, 1, 0.3), (10, 2, 0.3), (10, 7, 1.0), (10, 8, 1.0), (10, 9, 1.0), (10, 10, 1.0)
    )

    new SARScala()
      .getItemSimilarity(f.data.apply("getItemSimilarityInput"))
      .orderBy("i1", "i2")
      .toDF()
      .collect()
      .zip(expected)
      .foreach({case(row, testValue) =>
        assert(row.getAs[Int]("i1") === testValue._1)
        assert(row.getAs[Int]("i2") === testValue._2)
        assert(math.abs(row.getAs[Double]("value") - testValue._3) < 0.1)
      })
  }

  it should "calculate item-item similarity - cooccur" in { f =>
    val expected = Seq(
      (1, 1, 3.0), (1, 2, 3.0), (1, 3, 2.0), (1, 4, 1.0), (1, 7, 1.0), (1, 8, 1.0), (1, 9, 1.0), (1, 10, 1.0),
      (2, 1, 3.0), (2, 2, 3.0), (2, 3, 2.0), (2, 4, 1.0), (2, 7, 1.0), (2, 8, 1.0), (2, 9, 1.0), (2, 10, 1.0),
      (3, 1, 2.0), (3, 2, 2.0), (3, 3, 2.0), (3, 4, 1.0),
      (4, 1, 1.0), (4, 2, 1.0), (4, 3, 1.0), (4, 4, 1.0),
      (7, 1, 1.0), (7, 2, 1.0), (7, 7, 1.0), (7, 8, 1.0), (7, 9, 1.0), (7, 10, 1.0),
      (8, 1, 1.0), (8, 2, 1.0), (8, 7, 1.0), (8, 8, 1.0), (8, 9, 1.0), (8, 10, 1.0),
      (9, 1, 1.0), (9, 2, 1.0), (9, 7, 1.0), (9, 8, 1.0), (9, 9, 1.0), (9, 10, 1.0),
      (10, 1, 1.0), (10, 2, 1.0), (10, 7, 1.0), (10, 8, 1.0), (10, 9, 1.0), (10, 10, 1.0)
    )

    new SARScala()
      .setSimilarityMetric("cooccur")
      .getItemSimilarity(f.data.apply("getItemSimilarityInput"))
      .orderBy("i1", "i2")
      .toDF()
      .collect()
      .zip(expected)
      .foreach({case(row, testValue) =>
        assert(row.getAs[Int]("i1") === testValue._1)
        assert(row.getAs[Int]("i2") === testValue._2)
        assert(math.abs(row.getAs[Double]("value") - testValue._3) < 0.1)
      })
  }

  it should "calculate item-item similarity - lift" in { f =>
    val expected = Seq(
      (1, 1, 0.3), (1, 2, 0.3), (1, 3, 0.3), (1, 4, 0.3), (1, 7, 0.3), (1, 8, 0.3), (1, 9, 0.3), (1, 10, 0.3),
      (2, 1, 0.3), (2, 2, 0.3), (2, 3, 0.3), (2, 4, 0.3), (2, 7, 0.3), (2, 8, 0.3), (2, 9, 0.3), (2, 10, 0.3),
      (3, 1, 0.3), (3, 2, 0.3), (3, 3, 0.5), (3, 4, 0.5),
      (4, 1, 0.3), (4, 2, 0.3), (4, 3, 0.5), (4, 4, 1.0),
      (7, 1, 0.3), (7, 2, 0.3), (7, 7, 1.0), (7, 8, 1.0), (7, 9, 1.0), (7, 10, 1.0),
      (8, 1, 0.3), (8, 2, 0.3), (8, 7, 1.0), (8, 8, 1.0), (8, 9, 1.0), (8, 10, 1.0),
      (9, 1, 0.3), (9, 2, 0.3), (9, 7, 1.0), (9, 8, 1.0), (9, 9, 1.0), (9, 10, 1.0),
      (10, 1, 0.3), (10, 2, 0.3), (10, 7, 1.0), (10, 8, 1.0), (10, 9, 1.0), (10, 10, 1.0)
    )

    new SARScala()
      .setSimilarityMetric("lift")
      .getItemSimilarity(f.data.apply("getItemSimilarityInput"))
      .orderBy("i1", "i2")
      .toDF()
      .collect()
      .zip(expected)
      .foreach({case(row, testValue) =>
        assert(row.getAs[Int]("i1") === testValue._1)
        assert(row.getAs[Int]("i2") === testValue._2)
        assert(math.abs(row.getAs[Double]("value") - testValue._3) < 0.1)
      })
  }

  it should "get mapped arrays" in { f =>

    val expectedCounts = Array(8, 8, 4, 4, 6, 6, 6, 6)
    val expectedIndices = Array(
      0, 1, 2, 3, 4, 5, 6, 7,
      0, 1, 2, 3, 4, 5, 6, 7,
      0, 1, 2, 3,
      0, 1, 2, 3,
      0, 1, 4, 5, 6, 7,
      0, 1, 4, 5, 6, 7,
      0, 1, 4, 5, 6, 7,
      0, 1, 4, 5, 6, 7
    )
    val expectedValues = Array(
      1.0, 1.0, 0.333, 0.333, 0.333, 0.333, 0.333, 0.333,
      1.0, 1.0, 0.333, 0.333, 0.333, 0.333, 0.333, 0.333,
      0.333, 0.333, 1.0, 1.0,
      0.333, 0.333, 1.0, 1.0,
      0.333, 0.333, 1.0, 1.0, 1.0, 1.0,
      0.333, 0.333, 1.0, 1.0, 1.0, 1.0,
      0.333, 0.333, 1.0, 1.0, 1.0, 1.0,
      0.333, 0.333, 1.0, 1.0, 1.0, 1.0)

    val itemSimilarity = f.data.apply("itemSimilarity")
    val (counts, indices, values) = new SARScalaModel("uid_1", itemSimilarity, itemSimilarity).getMappedArrays

    assert(counts === expectedCounts)
    assert(indices === expectedIndices)
    assert(values === expectedValues)
  }

  it should "get processed ratings - no decay" in { f =>
    
    val expected = Seq(
      (1,1,1.0),
      (1,2,1.0),
      (1,3,1.0),
      (1,4,1.0),
      (2,1,1.0),
      (2,2,1.0),
      (2,7,1.0),
      (2,8,1.0),
      (2,9,1.0),
      (2,10,1.0),
      (3,1,1.0),
      (3,2,1.0),
      (3,3,1.0))
    
    new SARScala()
      .setUserCol("user")
      .setItemCol("item")
      .setRatingCol("rating")
      .setTimeCol("time")
      .getProcessedRatings(f.data.apply("getProcessedRatingsInput"))
      .orderBy("user", "item")
      .toDF()
      .collect()
      .zip(expected)
      .foreach({case(row, testValue) =>
        assert(row.getAs[Int]("user") === testValue._1)
        assert(row.getAs[Int]("item") === testValue._2)
        assert(math.abs(row.getAs[Double]("rating") - testValue._3) < 0.001)
      })
  }

  it should "get processed ratings - time decay" in { f =>

    // TODO: double check these values
    val expected = Seq(
      (1,1,0.007),
      (1,2,0.004),
      (1,3,0.008),
      (1,4,0.018),
      (2,1,0.004),
      (2,2,0.008),
      (2,7,0.002),
      (2,8,0.018),
      (2,9,0.040),
      (2,10,0.090),
      (3,1,0.201),
      (3,2,0.448),
      (3,3,1.0))

    new SARScala()
      .setUserCol("user")
      .setItemCol("item")
      .setRatingCol("rating")
      .setTimeCol("time")
      .setTimeDecay(true)
      .setDecayCoefficient(0.00001)
      .getProcessedRatings(f.data.apply("getProcessedRatingsInput"))
      .orderBy("user", "item")
      .toDF()
      .collect()
      .zip(expected)
      .foreach({case(row, testValue) =>
        println(row)
        assert(row.getAs[Int]("user") === testValue._1)
        assert(row.getAs[Int]("item") === testValue._2)
        assert(math.abs(row.getAs[Double]("rating") - testValue._3) < 0.001)
      })
  }

  it should "get user affinity" in { f =>

    val expected = Seq(
      (1,1,0.007),
      (1,2,0.004),
      (1,3,0.008),
      (1,4,0.018),
      (2,1,0.004),
      (2,2,0.008),
      (2,7,0.002),
      (2,8,0.018),
      (2,9,0.040),
      (2,10,0.090))

    new SARScalaModel("uid", f.data.apply("itemSimilarity"), f.data.apply("processedRatings"))
      .setUserCol("user")
      .setItemCol("item")
      .setRatingCol("rating")
      .getUserAffinity(f.data.apply("getUserAffinityInput"))
      .orderBy("user", "item")
      .toDF()
      .collect()
      .zip(expected)
      .foreach({case(row, testValue) =>
        assert(row.getAs[Int]("user") === testValue._1)
        assert(row.getAs[Int]("item") === testValue._2)
        assert(math.abs(row.getAs[Double]("rating") - testValue._3) < 0.1)
      })
  }}