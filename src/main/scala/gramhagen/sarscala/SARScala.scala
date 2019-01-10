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

import scala.collection.mutable

case class UserAffinity(u1: Integer, i1: Integer, score: Float)

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
  * @param userAffinity user affinity matrix
  */
class SARScalaModel (
  override val uid: String,
  @transient val itemSimilarity: Dataset[_],
  @transient val userAffinity: DataFrame)
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

  def getMappedArrays: (Array[Long], Array[Int], Array[Double]) = {

    val itemCounts = itemSimilarity.groupBy("i1")
      .count()
      .orderBy("i1")
      .collect()
      .map((r: Row) => r.getAs[Long]("count"))

    val itemMapping = itemSimilarity.select(col("i2"))
      .distinct()
      .select(col("i2"), (f.row_number().over(Window.orderBy("i2")) - 1).as("idx"))
      .repartition(col("i2"))
      .sortWithinPartitions()

    val itemIdsBuffer = new mutable.ArrayBuilder.ofInt
    val itemValuesBuffer = new mutable.ArrayBuilder.ofDouble
    itemSimilarity.join(itemMapping, "i2")
      .select(col("i1"), col("idx").as("i2"), col("value"))
      .orderBy("i1")
      .collect()
      .foreach((r: Row) => {
        itemIdsBuffer += r.getAs[Int]("i2")
        itemValuesBuffer += r.getAs[Double]("value")
    })

    (itemCounts,
     itemIdsBuffer.result,
     itemValuesBuffer.result)
  }


  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema)

    val (itemCounts, itemIds, itemValues) = getMappedArrays

    println(itemCounts.deep.mkString(","))

    userAffinity.show()

    val schema = Seq(
           StructField("u1", IntegerType, nullable = false),
           StructField("i1", IntegerType, nullable = false),
           StructField("itemCol", DoubleType, nullable = false))

    import userAffinity.sqlContext.implicits._

    // var encoder = RowEncoder(StructType(schema))

    val result = userAffinity
      //.map((r: Row) => UserAffinity(r.getInt(0), r.getInt(1), r.getFloat(2)))
      .groupByKey(r => r.getInt(0)) // TODO: 0 is not ideal
      .flatMapGroups((u1, rowsForEach) => {
        val list1 = scala.collection.mutable.ListBuffer[UserAffinity]()
        list1.append(UserAffinity(u1,2,1.2f))
        // Iterable(UserAffinity(1,2,1.2f))
        list1
      })
      .toDF
      //.groupByKey((ua: Row) => ua.u1)
      //.flatMapGroups((u1:Integer, rowsForEach) => {
      //  val list1 = scala.collection.mutable.ListBuffer[Row]()
      // list1
      //}, encoder)
      //.toDF
      // .groupBy(col("u1"))
      //.groupByKey(_.get(0)) // TODO: 0 is not ideal
      // .groupByKey(_ => 1) // TODO: 0 is not ideal 
      /*
      .flatMapGroups((u1, rowsForEach) => {
        val list1 = scala.collection.mutable.ListBuffer[Row]()
        // for (elem <- rowsForEach) {
         //  list1.append(Row(1, 2, 1.2))
        // }
        // list1
        // var list1:List[Row] = List()
        // list1 :+ Row(1, 2, 1.2)
        list1
      }, encoder).toDF
*/
    //println(result.schema)

    result.show()
/*
 def recommend() : Any = {
        // parameters
        val itemsOfUsers // TODO: vector<int32_t> type -> array?
        val ratings // TODO: vector<float> 
        val top_k = 10 // TODO: method vs global parameter?
        val remove_seen = true // TODO: method vs global parameter?

        // TODO handle empty users
        
        // TODO sort itemsOfUsers + ratings in parallel by itemId
        val sortedItemIdsAndRatings;

        val seen_items // TODO: hash_set<int32_t>
        if (remove_seen) {
            // TODO: add all itemsOfUser to seen_items
        }

        val top_k_items // TODO: priority_queue<item, score> sorted desc by score        

        // TODO: loop syntax?
        for itemId : itemsOfUser  {

            relatedIdxBegin = offsets[itemId]
            relatedIdxEnd = offsets[itemId + 1]

            // loop through items user has seen
            for (val i = relatedIdxBegin; i<relatedIdxEnd; i++) {

                val relatedItemId = related[i]

                if (top_k_items.contains(relatedItemId))
                    continue
                
                top_k_items.add(relatedItemId)

                val relatedItemScore = joinProdSum(sortedItemIdsAndRatings, relatedItemId)

                if (relatedItemScore > 0)
                    pushIfBetter(top_k_items, relatedItemId, relatedItemScore, top_k)
            }
        }
        
        // TODO: empty the top_k_items
        return predictions
    }

    // TODO: private
    def pushIfBetter() : Float = {
        if (topKItems.size() < topK)
            topKItems.add(itemIdAndScore)

        if (topKItems.top().score < itemIdAndScore.score) {
            topKItems.pop()
            topKItems.push(itemIdAndScore)
        }
    }

    def joinProdSum(relatedItemId) : Float = {
        val contribIdxBegin = offset[relatedItemId]
        val contribIdxEnd = offset[relatedItemId + 1]

        val score = 0.;

        val userIndex
    }*/
    // TODO: add scoring here!
    dataset.select(dataset("*"))
  }

  override def transformSchema(schema: StructType): StructType = {
    // append prediction column
    StructType(schema.fields :+ StructField($(predictionCol), IntegerType, nullable = false))
  }

  override def copy(extra: ParamMap): SARScalaModel = {
    new SARScalaModel(uid, itemSimilarity, userAffinity)
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

       val userAffinityPath = new Path(path, "userAffinity").toString
       instance.userAffinity.write.format("parquet").save(userAffinityPath)
    }
  }

  private class SARScalaModelReader extends MLReader[SARScalaModel] {

    override def load(path: String): SARScalaModel = {
      val metadataPath = new Path(path, "metadata").toString
      val metadata = sparkSession.read.format("parquet").load(metadataPath).first()

      val itemSimilarityPath = new Path(path, "itemSimilarity").toString
      val itemSimilarity = sparkSession.read.format("parquet").load(itemSimilarityPath)

      val userAffinityPath = new Path(path, "userAffinity").toString
      val userAffinity = sparkSession.read.format("parquet").load(userAffinityPath)

      new SARScalaModel(metadata.getAs[String]("uid"), itemSimilarity, userAffinity)
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

  def getUserAffinity(df: Dataset[_]): DataFrame = {
    val user = df.filter(col($(ratingCol)) > 0)
      .select(col($(userCol)).as("u1"), col($(itemCol)).as("i1"))

    // TODO: change this to original implementation
    user.join(user.select(col("u1"), col("i1")), "i1")
      .groupBy(col("u1"), col("i1"))
      .agg(f.sum(f.lit(1.0)).as("value"))
      .repartition(col("u1"), col("i1"))
      .sortWithinPartitions()
  }

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
      case "cooccur" =>
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

  def getProcessedRatings(df: Dataset[_]): Dataset[_] = {

    if ($(timeDecay)) {
      val latest = df.select(f.max($(timeCol))).first().get(0)
      val decay = -math.log(2) / ($(decayCoefficient) * 60 * 60 * 24)

      df.groupBy($(userCol), $(itemCol))
        .agg(f.sum(col($(ratingCol)) * f.exp(f.lit(decay) * (f.lit(latest) - col($(timeCol))))).as($(ratingCol)))
        .repartition(col($(userCol)))
        .sortWithinPartitions()
    } else {
      df.select(col($(userCol)), col($(itemCol)), col($(ratingCol)).cast(DoubleType),
        f.row_number().over(Window.partitionBy($(userCol), $(itemCol)).orderBy(f.desc($(timeCol)))).as("latest"))
        .filter(col("latest") === 1)
        .drop("latest")
    }
  }

  override def fit(dataset: Dataset[_]): SARScalaModel = {

    // apply time decay to ratings if necessary otherwise remove duplicates
    val processedRatings = getProcessedRatings(dataset)

    // calculate user affinity
    val userAffinity = getUserAffinity(processedRatings)

    // count item-item co-occurrence
    val itemCoOccurrence = getItemCoOccurrence(processedRatings)

    // calculate item-item similarity
    val itemSimilarity = getItemSimilarity(itemCoOccurrence)

    new SARScalaModel(uid, itemSimilarity, userAffinity)
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