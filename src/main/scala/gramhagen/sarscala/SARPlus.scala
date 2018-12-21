package gramhagen.sarscala

import org.apache.hadoop.fs.Path
import org.apache.spark.ml.param.shared.HasPredictionCol
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset, Row}

import scala.collection.mutable


/**
  * Common params for SARPlus Model.
  */
trait SARPlusModelParams extends Params with HasPredictionCol {

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
class SARPlusModel (
  override val uid: String,
  @transient val itemSimilarity: DataFrame)
  extends Model[SARPlusModel] with SARPlusModelParams with MLWritable {

  /** @group setParam */
  def setUserCol(value: String): this.type = set(userCol, value)

  /** @group setParam */
  def setItemCol(value: String): this.type = set(itemCol, value)

  /** @group setParam */
  def setRatingCol(value: String): this.type = set(ratingCol, value)

  /** @group setParam */
  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  def getMappedArrays: Map[String, Array[_]] = {
    val itemCountsBuffer = new mutable.ArrayBuilder.ofLong
    val itemIdsBuffer = new mutable.ArrayBuilder.ofInt
    val itemValuesBuffer = new mutable.ArrayBuilder.ofDouble

    itemSimilarity.groupBy("i1")
      .count()
      .orderBy(("i1"))
      .foreach((r: Row) => {
        itemCountsBuffer += r.getAs[Long]("count")
    })

    itemSimilarity.orderBy(col("i1"))
      .foreach((r: Row) => {
        itemIdsBuffer += r.getAs[Int]("i2")
        itemValuesBuffer += r.getAs[Double]("value")
    })

    Map("itemSimilarityCounts" -> itemCountsBuffer.result,
        "itemSimilarityIds" -> itemIdsBuffer.result,
        "itemSimilarityValues" -> itemValuesBuffer.result)
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema)
    dataset.select(dataset("*"))
  }

  override def transformSchema(schema: StructType): StructType = {
    // append prediction column
    StructType(schema.fields :+ StructField($(predictionCol), IntegerType, nullable = false))
  }

  override def copy(extra: ParamMap): SARPlusModel = {
    val copied = new SARPlusModel(uid, itemSimilarity)
    copyValues(copied, extra).setParent(parent)
  }

  override def write: MLWriter = new SARPlusModel.SARPlusModelWriter(this)
}

object SARPlusModel extends MLReadable[SARPlusModel] {

  override def read: MLReader[SARPlusModel] = new SARPlusModelReader

  override def load(path: String): SARPlusModel = super.load(path)

  private[SARPlusModel] class SARPlusModelWriter(instance: SARPlusModel) extends MLWriter {

     override protected def saveImpl(path: String): Unit = {
       // FIXM: store uid properly
       // TODO: extend this to include more metadata
       val itemSimilarityPath = new Path(path, "itemSimilarity").toString
       instance.itemSimilarity.write.format("parquet").save(itemSimilarityPath)
    }
  }

  private class SARPlusModelReader extends MLReader[SARPlusModel] {

    override def load(path: String): SARPlusModel = {
      // TODO: extend this to include more metadata
      val uidPath = new Path(path, "uid").toString
      val uid = sparkSession.read.format("string").load(uidPath).toString

      val itemSimilarityPath = new Path(path, "itemSimilarity").toString
      val itemSimilarity = sparkSession.read.format("parquet").load(itemSimilarityPath)

      new SARPlusModel(uid, itemSimilarity)
    }
  }
}

class SARPlus (override val uid: String) extends Estimator[SARPlusModel] with SARPlusModelParams {

  def this() = this(Identifiable.randomUID("sarplus"))

  /** @group setParam */
  def setUserCol(value: String): this.type = set(userCol, value)

  /** @group setParam */
  def setItemCol(value: String): this.type = set(itemCol, value)

  /** @group setParam */
  def setRatingCol(value: String): this.type = set(ratingCol, value)

  /** @group setParam */
  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  override def fit(dataset: Dataset[_]): SARPlusModel = {

    // first we count item-item co-occurrence
    val dfA = dataset.as("dfA")
    val dfB = dataset.as("dfB")
    val itemCooccurrence = dfA.join(dfB, dfA.col($(userCol)) === dfB.col($(userCol)) && dfA.col($(itemCol)) === dfB.col($(itemCol)))
      .groupBy(dfA.col($(itemCol)), dfB.col($(itemCol)))
      .count()
      .filter(col("count") > 0)
      .select(dfA.col($(itemCol)).as("i1"), dfB.col($(itemCol)).as("i2"), col("count"))
      .repartition(col("i1"), col("i2"))
      .sortWithinPartitions()

    // next we count each item occurrence
    val itemMarginal = itemCooccurrence.filter(col("i1") === col("i2"))
      .select(col("i1").as("i"), col("count"))

    val dfIC = itemCooccurrence.as("dfIC")
    val dfM = itemMarginal.as("dfM")

    // compute the Jaccard distance between items, this is symmetric so only compute the upper triangular
    val dfICM = dfIC.join(dfM, dfIC.col("i1")   === dfM.col("i"))
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

    new SARPlusModel(uid, itemSimilarity)
  }

  override def transformSchema(schema: StructType): StructType = {
    transformSchema(schema)
  }

  override def copy(extra: ParamMap): SARPlus = defaultCopy(extra)
}