package gramhagen.sarscala

import gramhagen.sarscala.SARExtensions._
import org.scalatest._

class SARScalaPredictorSpec extends FlatSpec {

    "predictor" should "return index" in  {
        // similarity matrix
        // 0-1, 1,2
        // 0,3,5
        // 0,1, 0,1,2, 1,2 
        
        val itemOffsets = Array(0,2,5,7)
        val itemIds = Array(0,1,0,1,2,1,2)
        val itemValues = Array(0.1f,0.2f,0.1f,0.2f,0.3f,0.3f,0.4f)

        val userRatings = Array(
            ItemScore(0, 1),
            ItemScore(2, 2)
        )

        val predictions = new SARScalaPredictor(
            itemOffsets,
            itemIds,
            itemValues).predict(25, userRatings, 10)

        for (p <- predictions) {
            // println(s"$p.u1 to $p.i1")
            println(p)
        }
    }
}