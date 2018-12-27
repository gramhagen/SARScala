package gramhagen.sarscala

import org.apache.hadoop.fs.Path
import org.apache.spark.ml.param.shared.HasPredictionCol
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.{col, row_number}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset, Row}

import scala.collection.mutable


/**
  * Common params for SARScala Model.
  */
trait SARScalaModelParams extends Params with HasPredictionCol {

  /** @group param */
  val userCol = new Param[String](this, "userCol", "column name for user ids. all ids must be integers.")

  /** @group param */
  def getUserCol: String = $(userCol)

  /** @group param */
  val itemCol = new Param[String](this, "itemCol", "column name for item ids. all ids must be integers.")

  /** @group param */
  def getItemCol: String = $(itemCol)

  /** @group param */
  val ratingCol = new Param[String](this, "ratingCol", "column name for ratings.")

  /** @group param */
  def getRatingCol: String = $(ratingCol)
}

/**
  * Model fitted by SAR
  *
  * @param itemSimilarity continuous index-mapped item similarity matrix
  */
class SARScalaModel (
  override val uid: String,
  @transient val itemSimilarity: DataFrame)
  extends Model[SARScalaModel] with SARScalaModelParams with MLWritable {

  /** @group setParam */
  def setUserCol(value: String): this.type = set(userCol, value)

  /** @group setParam */
  def setItemCol(value: String): this.type = set(itemCol, value)

  /** @group setParam */
  def setRatingCol(value: String): this.type = set(ratingCol, value)

  /** @group setParam */
  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  def getMappedArrays: (Array[Long], Array[Int], Array[Double]) = {

    val itemCountsBuffer = new mutable.ArrayBuilder.ofLong
    itemSimilarity.groupBy(col("i1"))
      .count()
      .orderBy(col("i1"))
      .collect()
      .foreach((r: Row) => {
        itemCountsBuffer += r.getAs[Long]("count")
      })

    val itemMapping = itemSimilarity.select("i1")
      .distinct()
      .select(col("i1").as("i"),
        (row_number().over(Window.orderBy(col("i1"))) - 1).as("idx"))
      .repartition(col("i"))
      .sortWithinPartitions()

    val dfIS = itemSimilarity.as("dfIS")
    val dfIM = itemMapping.as("dfIM")

    val itemSimilarityMapped = dfIS.join(dfIM, dfIS.col("i2") === dfIM.col("i"))
      .select(col("i1"), col("idx").as("i2"), col("value"))
      .cache

    val itemIdsBuffer = new mutable.ArrayBuilder.ofInt
    val itemValuesBuffer = new mutable.ArrayBuilder.ofDouble
    itemSimilarityMapped.orderBy(col("i1"))
      .collect()
      .foreach((r: Row) => {
        itemIdsBuffer += r.getAs[Int]("i2")
        itemValuesBuffer += r.getAs[Double]("value")
    })

    (itemCountsBuffer.result,
     itemIdsBuffer.result,
     itemValuesBuffer.result)
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema)
    dataset.select(dataset("*"))
  }

  override def transformSchema(schema: StructType): StructType = {
    // append prediction column
    StructType(schema.fields :+ StructField($(predictionCol), IntegerType, nullable = false))
  }

  override def copy(extra: ParamMap): SARScalaModel = {
    val copied = new SARScalaModel(uid, itemSimilarity)
    copyValues(copied, extra).setParent(parent)
  }

  override def write: MLWriter = new SARScalaModel.SARScalaModelWriter(this)
}

object SARScalaModel extends MLReadable[SARScalaModel] {

  override def read: MLReader[SARScalaModel] = new SARScalaModelReader

  override def load(path: String): SARScalaModel = super.load(path)

  private[SARScalaModel] class SARScalaModelWriter(instance: SARScalaModel) extends MLWriter {

     override protected def saveImpl(path: String): Unit = {
       // FIXM: store uid properly
       // TODO: extend this to include more metadata
       val itemSimilarityPath = new Path(path, "itemSimilarity").toString
       instance.itemSimilarity.write.format("parquet").save(itemSimilarityPath)
    }
  }

  private class SARScalaModelReader extends MLReader[SARScalaModel] {

    override def load(path: String): SARScalaModel = {
      // TODO: extend this to include more metadata
      val uidPath = new Path(path, "uid").toString
      val uid = sparkSession.read.format("string").load(uidPath).toString

      val itemSimilarityPath = new Path(path, "itemSimilarity").toString
      val itemSimilarity = sparkSession.read.format("parquet").load(itemSimilarityPath)

      new SARScalaModel(uid, itemSimilarity)
    }
  }
}

class SARScala (override val uid: String) extends Estimator[SARScalaModel] with SARScalaModelParams {

  def this() = this(Identifiable.randomUID("SARScala"))

  /** @group setParam */
  def setUserCol(value: String): this.type = set(userCol, value)

  /** @group setParam */
  def setItemCol(value: String): this.type = set(itemCol, value)

  /** @group setParam */
  def setRatingCol(value: String): this.type = set(ratingCol, value)

  /** @group setParam */
  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  override def fit(dataset: Dataset[_]): SARScalaModel = {

    // first we count item-item co-occurrence
    val dfA = dataset.select(col($(userCol)).as("u1"), col($(itemCol)).as("i1"))
    val dfB = dataset.select(col($(userCol)).as("u2"), col($(itemCol)).as("i2"))

    val itemCooccurrence = dfA.join(dfB,
         col("u1") <=> col("u2") && // take care of nulls w/ <=>
         col("i1") <= col("i2"))
      .groupBy(col("i1"), col("i2"))
      .count()
      .filter(col("count") > 0) // TODO: implement threshold
      .repartition(col("i1"), col("i2"))
      .sortWithinPartitions()

    // next we count each item occurrence
    val itemMarginal = itemCooccurrence.filter(col("i1") === col("i2"))
      .select(col("i1").as("i"), col("count"))

    val dfIC = itemCooccurrence.as("dfIC")
    val dfM = itemMarginal.as("dfM")

    // compute the Jaccard distance between items, this is symmetric so only compute the upper triangular
    val dfICM = dfIC.join(dfM, dfIC.col("i1") === dfM.col("i"))
      .select(dfIC.col("*"),
              (dfM.col("count") - dfIC.col("count")).as("i1_marginal"))

    val itemSimilarityUpper = dfICM.join(dfM, dfICM.col("i2") === dfM.col("i"))
      .select(dfICM.col("i1"),
              dfICM.col("i2"),
              (dfICM.col("count") / (dfICM.col("i1_marginal") + dfM.col("count"))).as("value"))

    // fill in the lower triangular
    val itemSimilarity = itemSimilarityUpper.union(
        itemSimilarityUpper.filter(col("i1") =!= col("i2"))
                           .select(col("i2"), col("i1"), col("value")))
      .repartition(col("i1"))
      .sortWithinPartitions()

    new SARScalaModel(uid, itemSimilarity)
  }

  override def transformSchema(schema: StructType): StructType = {
    transformSchema(schema)
  }

  override def copy(extra: ParamMap): SARScala = defaultCopy(extra)
}