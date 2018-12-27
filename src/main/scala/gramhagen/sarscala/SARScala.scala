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
  @transient val itemSimilarity: Dataset[_])
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

    val itemCounts = itemSimilarity.groupBy("i1")
      .count()
      .orderBy("i1")
      .collect()
      .map((r: Row) => r.getAs[Long]("count"))

    val itemMapping = itemSimilarity.select(col("i1").as("i"))
      .distinct()
      .select(col("i"),
        (row_number().over(Window.orderBy(col("i"))) - 1).as("idx"))
      .repartition(col("i"))
      .sortWithinPartitions()

    val itemIdsBuffer = new mutable.ArrayBuilder.ofInt
    val itemValuesBuffer = new mutable.ArrayBuilder.ofDouble
    itemSimilarity.join(itemMapping, col("i2") === col("i"))
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

  def getItemCoOcurrence(df: Dataset[_]): Dataset[_] = {
    df.select(col($(userCol)).as("u1"), col($(itemCol)).as("i1"))
      .join(df.select(col($(userCol)).as("u2"), col($(itemCol)).as("i2")),
        col("u1") <=> col("u2") && // remove nulls with <=>
        col("i1") <= col("i2"))
      .groupBy(col("i1"), col("i2"))
      .count()
      .filter(col("count") > 0) // TODO: implement threshold
      .repartition(col("i1"), col("i2"))
      .sortWithinPartitions()
  }

  def getItemSimilarity(df: Dataset[_], metric: String): Dataset[_] = {
    // count each item occurrence
    val itemCount = df.filter(col("i1") === col("i2"))

    // compute upper triangular of the item-item similarity matrix using Jaccard distance between items
    val upperTriangular = df.join(itemCount.select(col("i1"), col("count").as("i1_count")), "i1")
      .select(col("i1"), col("i2"), col("count"),
        (col("i1_count") - col("count")).as("i1_marginal"))
      .join(itemCount.select(col("i2"), col("count").as("i2_count")), "i2")
      .select(col("i1"), col("i2"),
        (col("count") / (col("i1_marginal") + col("i2_count"))).as("value"))

    // fill in the lower triangular
    upperTriangular.union(
      upperTriangular.filter(col("i1") =!= col("i2"))
        .select(col("i2"), col("i1"), col("value")))
      .repartition(col("i1"))
      .sortWithinPartitions()
  }

  override def fit(dataset: Dataset[_]): SARScalaModel = {

    // first we count item-item co-occurrence
    val itemCoOccurrence = getItemCoOcurrence(dataset)

    val itemSimilarity = getItemSimilarity(itemCoOccurrence, "jaccard")

    new SARScalaModel(uid, itemSimilarity)
  }

  override def transformSchema(schema: StructType): StructType = {
    transformSchema(schema)
  }

  override def copy(extra: ParamMap): SARScala = defaultCopy(extra)
}