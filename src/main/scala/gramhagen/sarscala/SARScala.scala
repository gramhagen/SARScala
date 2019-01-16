package gramhagen.sarscala

import org.apache.hadoop.fs.Path
import org.apache.spark.ml.param.shared.HasPredictionCol
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.{functions => f}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.sql.catalyst.encoders.RowEncoder

import scala.collection.mutable.{ArrayBuilder, HashSet, ListBuffer, PriorityQueue}
import util.control.Breaks._
import gramhagen.sarscala.SARExtensions._

case class UserAffinity(u1: Int, i1: Int, score: Float)

case class ItemScore(id: Int, score: Float) extends Ordered[ItemScore] {
  // Required as of Scala 2.11 for reasons unknown - the companion to Ordered
  // should already be in implicit scope
  import scala.math.Ordered.orderingToOrdered

  def compare(that: ItemScore): Int = this.score compare that.score
}

/**
  * Common params for SARScala Model.
  */
trait SARScalaModelParams extends Params with HasPredictionCol {

  /** @group param */
  val userCol = new Param[String](this, "userCol", "column name for user ids, all ids must be integers")

  /** @group param */
  def getUserCol: String = $(userCol)

  /** @group param */
  val itemCol = new Param[String](this, "itemCol", "column name for item ids, all ids must be integers")

  /** @group param */
  def getItemCol: String = $(itemCol)

  /** @group param */
  val ratingCol = new Param[String](this, "ratingCol", "column name for ratings")

  /** @group param */
  def getRatingCol: String = $(ratingCol)

  /** @group param */
  val timeCol = new Param[String](this, "timeCol", "column name for timestamps, all timestamps must be longs")

  /** @group param */
  def getTimeCol: String = $(timeCol)
}

/**
  * Common parameters for SARScala algorithm
  */
trait SARScalaParams extends SARScalaModelParams {

  /** @group param */
  val timeDecay = new Param[Boolean](this, "timeDecay", "flag to enable time decay on ratings")

  /** @group param */
  def getTimeDecay: Boolean = $(timeDecay)

  /** @group param */
  val decayCoefficient = new Param[Double](this, "decayCoefficient", "time decay coefficient, number of days for rating to decay by half")

  /** @group param */
  def getDecayCoefficient: Double = $(decayCoefficient)

  /** @group param */
  val similarityMetric = new Param[String](this, "similarityMetric", "metric to use for item-item similarity, must one of: [ cooccur | jaccard | lift ]]")

  /** @group param */
  def getSimilarityMetric: String = $(similarityMetric)

  /** @group param */
  val countThreshold = new Param[Int](this, "countThreshold", "ignore item co-occurrences that are less than the threshold")

  /** @group param */
  def getCountThreshold: Int = $(countThreshold)
}

/**
  * Model fitted by SAR
  *
  * @param itemSimilarity item similarity matrix
  * @param processedRatings user affinity matrix
  */
class SARScalaModel (
  override val uid: String,
  @transient val itemSimilarity: Dataset[_],
  @transient val processedRatings: DataFrame)
  extends Model[SARScalaModel] with SARScalaModelParams with MLWritable {

  /** @group setParam */
  def setUserCol(value: String): this.type = set(userCol, value)

  /** @group setParam */
  def setItemCol(value: String): this.type = set(itemCol, value)

  /** @group setParam */
  def setRatingCol(value: String): this.type = set(ratingCol, value)

  /** @group setParam */
  def setTimeCol(value: String): this.type = set(timeCol, value)

  /** @group setParam */
  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  def getUserAffinity(test_df: Dataset[_]): Dataset[_] = {
    test_df.filter(col($(ratingCol)) > 0)
      .select(col($(userCol)))
      .distinct()
      .join(processedRatings, Seq($(userCol)))
      .select(col($(userCol)), col($(itemCol)), col($(ratingCol)))
      .repartition(col($(userCol)), col($(itemCol)))
      .sortWithinPartitions()
  }

  def getMappedArrays: (DataFrame, Array[Long], Array[Int], Array[Double]) = {

    val itemOffsets = 
      itemSimilarity.groupBy("i1")
      .count()
      .orderBy("i1")
      .select("count") // reduce amount transferred
      .collect()
      .map((r: Row) => r.getAs[Long]("count"))
      .scanLeft(0L)(_+_) // cumulative sum to get to offsets

    val itemMapping = itemSimilarity.select(col("i2"))
      .distinct()
      .select(col("i2"), (f.row_number().over(Window.orderBy("i2")) - 1).as("idx"))
      .repartition(col("i2"))
      .sortWithinPartitions()

    val itemIdsBuffer = new ArrayBuilder.ofInt
    val itemValuesBuffer = new ArrayBuilder.ofDouble
    itemSimilarity.join(itemMapping, "i2")
      .select(col("i1"), col("idx").as("i2"), col("value"))
      .orderBy("i1")
      .collect()
      .foreach((r: Row) => {
        itemIdsBuffer += r.getAs[Int]("i2")
        itemValuesBuffer += r.getAs[Double]("value")
    })

    (itemMapping,
     itemOffsets,
     itemIdsBuffer.result,
     itemValuesBuffer.result)
  }


  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema)

    val (itemMapping, itemOffsets, itemIds, itemValues) = getMappedArrays

    // println(itemOffsets.deep.mkString(","))
    // processedRatings.show()
/*
    val schema = Seq(
           StructField("u1", IntegerType, nullable = false),
           StructField("i1", IntegerType, nullable = false),
           StructField("itemCol", DoubleType, nullable = false))
*/
    import processedRatings.sqlContext.implicits._

/*
    val colName = $(itemCol)
    println(s"processedRatings: $colName")
    processedRatings.show()
*/
    val result = processedRatings
      .join(itemMapping, processedRatings.col($(itemCol)) <=> itemMapping.col("i2"))
      .select(col($(userCol)), col("idx"), col($(ratingCol)).cast(FloatType))
      .groupByKey(r => r.getInt(0)) // by user id
      .flatMapGroups((u1, rowsForEach) => {
        // TODO: switch to 2x array allocation instead of individual 
        val userRatings = rowsForEach
          .map(r => ItemScore(r.getInt(1), r.getFloat(2)))
            .toArray
            .sortWith(_.id < _.id)

        new SARScalaPredictor(itemOffsets, itemIds, itemValues)
          .predict(u1, userRatings)
      })
      .toDF

    // map back to original item space
    // rename columns
    result.join(itemMapping, result.col("i1") <=> itemMapping.col("idx"))
      .select(col("u1").as($(userCol)), col("i2").as($(itemCol)), col("score"))
  }

  override def transformSchema(schema: StructType): StructType = {
    // append prediction column
    StructType(schema.fields :+ StructField($(predictionCol), IntegerType, nullable = false))
  }

  override def copy(extra: ParamMap): SARScalaModel = {
    new SARScalaModel(uid, itemSimilarity, processedRatings)
      .setUserCol(getUserCol)
      .setItemCol(getItemCol)
      .setRatingCol(getRatingCol)
      .setPredictionCol(getPredictionCol)
  }

  override def write: MLWriter = new SARScalaModel.SARScalaModelWriter(this)
}

object SARScalaModel extends MLReadable[SARScalaModel] {

  override def read: MLReader[SARScalaModel] = new SARScalaModelReader

  override def load(path: String): SARScalaModel = super.load(path)

  private[SARScalaModel] class SARScalaModelWriter(instance: SARScalaModel) extends MLWriter {

     override protected def saveImpl(path: String): Unit = {

       val metadataPath = new Path(path, "metadata").toString

       val metadata = Seq(Row(
         instance.uid,
         instance.userCol,
         instance.itemCol,
         instance.ratingCol,
         instance.timeCol,
         instance.predictionCol))

       val schema = Seq(
           StructField("uid", StringType, nullable = false),
           StructField("userCol", StringType, nullable = false),
           StructField("itemCol", StringType, nullable = false),
           StructField("ratingCol", StringType, nullable = false),
           StructField("timeCol", StringType, nullable = true),
           StructField("predictionCol", StringType, nullable = true))

       sparkSession.createDataFrame(sparkSession.sparkContext.parallelize(metadata), StructType(schema))
         .write.format("parquet").save(metadataPath)

       val itemSimilarityPath = new Path(path, "itemSimilarity").toString
       instance.itemSimilarity.write.format("parquet").save(itemSimilarityPath)

       val processedRatingsPath = new Path(path, "processedRatings").toString
       instance.processedRatings.write.format("parquet").save(processedRatingsPath)
    }
  }

  private class SARScalaModelReader extends MLReader[SARScalaModel] {

    override def load(path: String): SARScalaModel = {
      val metadataPath = new Path(path, "metadata").toString
      val metadata = sparkSession.read.format("parquet").load(metadataPath).first()

      val itemSimilarityPath = new Path(path, "itemSimilarity").toString
      val itemSimilarity = sparkSession.read.format("parquet").load(itemSimilarityPath)

      val processedRatingsPath = new Path(path, "processedRatings").toString
      val processedRatings = sparkSession.read.format("parquet").load(processedRatingsPath)

      new SARScalaModel(metadata.getAs[String]("uid"), itemSimilarity, processedRatings)
        .setUserCol(metadata.getAs[String]("userCol"))
        .setItemCol(metadata.getAs[String]("itemCol"))
        .setRatingCol(metadata.getAs[String]("ratingCol"))
        .setTimeCol(metadata.getAs[String]("timeCol"))
        .setPredictionCol(metadata.getAs[String]("predictionCol"))
    }
  }
}

class SARScala (override val uid: String) extends Estimator[SARScalaModel] with SARScalaParams {

  // set default values
  this.setTimeDecay(false)
  this.setDecayCoefficient(30)
  this.setSimilarityMetric("jaccard")
  this.setCountThreshold(0)

  def this() = this(Identifiable.randomUID("SARScala"))

  /** @group setParam */
  def setUserCol(value: String): this.type = set(userCol, value)

  /** @group setParam */
  def setItemCol(value: String): this.type = set(itemCol, value)

  /** @group setParam */
  def setRatingCol(value: String): this.type = set(ratingCol, value)

  /** @group setParam */
  def setTimeCol(value: String): this.type = set(timeCol, value)

  /** @group setParam */
  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  /** @group setParam */
  def setTimeDecay(value: Boolean): this.type = set(timeDecay, value)

  /** @group setParam */
  def setDecayCoefficient(value: Double): this.type = set(decayCoefficient, value)

  /** @group setParam */
  def setSimilarityMetric(value: String): this.type = set(similarityMetric, value.toLowerCase)

  /** @group setParam */
  def setCountThreshold(value: Int): this.type = set(countThreshold, value)

  def getItemCoOccurrence(df: Dataset[_]): Dataset[_] = {
    df.select(col($(userCol)).as("u1"), col($(itemCol)).as("i1"))
      .join(df.select(col($(userCol)).as("u2"), col($(itemCol)).as("i2")),
        col("u1") <=> col("u2") && // remove nulls with <=>
        col("i1") <= col("i2"))
      .groupBy(col("i1"), col("i2"))
      .count()
      .filter(col("count") > $(countThreshold))
      .repartition(col("i1"), col("i2"))
      .sortWithinPartitions()
  }

  def getItemSimilarity(df: Dataset[_]): Dataset[_] = {

    // count each item occurrence
    val itemCount = df.filter(col("i1") === col("i2"))

    // append marginal occurrence counts for each item
    val itemMarginal = df.join(itemCount.select(col("i1"), col("count").as("i1_count")), "i1")
      .select(col("i1"), col("i2"), col("count"), col("i1_count"))
      .join(itemCount.select(col("i2"), col("count").as("i2_count")), "i2")

    // compute upper triangular of the item-item similarity matrix using desired metric between items
    val upperTriangular = $(similarityMetric) match {
      case "cooccurrence" =>
          df.select(col("i1"), col("i2"), col("count").cast(DoubleType).as("value"))
      case "jaccard" =>
        itemMarginal.select(col("i1"), col("i2"),
          (col("count") / (col("i1_count") + col("i2_count") - col("count"))).as("value"))
      case "lift" =>
        itemMarginal.select(col("i1"), col("i2"),
          (col("count") / (col("i1_count") * col("i2_count"))).as("value"))
      case _ =>
        throw new IllegalArgumentException("unsupported similarity metric")
    }

    // fill in the lower triangular
    upperTriangular.union(
      upperTriangular.filter(col("i1") =!= col("i2"))
        .select(col("i2"), col("i1"), col("value")))
      .repartition(col("i1"))
      .sortWithinPartitions()
  }

  def getProcessedRatings(df: Dataset[_]): DataFrame = {

    var dfResult:DataFrame = null

    if ($(timeDecay)) {
      val latest = df.select(f.max($(timeCol))).first().get(0)
      val decay = -math.log(2) / ($(decayCoefficient) * 60 * 60 * 24)

      dfResult = df.groupBy($(userCol), $(itemCol))
        .agg(f.sum(col($(ratingCol)) * f.exp(f.lit(decay) * (f.lit(latest) - col($(timeCol))))).as($(ratingCol)))
    } else {
      dfResult = df.select(col($(userCol)), col($(itemCol)), col($(ratingCol)).cast(DoubleType),
        f.row_number().over(Window.partitionBy($(userCol), $(itemCol)).orderBy(f.desc($(timeCol)))).as("latest"))
        .filter(col("latest") === 1)
        .drop("latest")
    }

    // important to sort as the scoring depends on it
    dfResult
        .repartition(col($(userCol)))
        .sortWithinPartitions()
  }

  override def fit(dataset: Dataset[_]): SARScalaModel = {

    // apply time decay to ratings if necessary otherwise remove duplicates
    val processedRatings = getProcessedRatings(dataset)

    // count item-item co-occurrence
    val itemCoOccurrence = getItemCoOccurrence(processedRatings)

    // calculate item-item similarity
    val itemSimilarity = getItemSimilarity(itemCoOccurrence)

    new SARScalaModel(uid, itemSimilarity, processedRatings)
      .setUserCol($(userCol))
      .setItemCol($(itemCol))
      .setRatingCol($(ratingCol))
      .setTimeCol($(timeCol))
  }

  override def transformSchema(schema: StructType): StructType = {
    transformSchema(schema)
  }

  override def copy(extra: ParamMap): SARScala = {
    copy(extra)
      .setTimeDecay(getTimeDecay)
      .setDecayCoefficient(getDecayCoefficient)
      .setSimilarityMetric(getSimilarityMetric)
      .setCountThreshold(getCountThreshold)
  }
}