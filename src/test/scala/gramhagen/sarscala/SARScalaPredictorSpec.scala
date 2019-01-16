package gramhagen.sarscala

import gramhagen.sarscala.SARExtensions._
import org.scalatest._

class SARScalaPredictorSpec extends FlatSpec {

    "predictor" should "return index" in  {
        // similarity matrix
        // 0-1, 1,2
        // 0,3,5
        // 0,1, 0,1,2, 1,2 
        
        val itemOffsets = Array(0L,2L,5L,7L)
        val itemIds = Array(0,1,0,1,2,1,2)
        val itemValues = Array(0.1,0.2,0.1,0.2,0.3,0.3,0.4)
        // assert(2 == Array(1,2,3).searchIndex(3, (a, b:Int) => a.compare(b)))

        val userRatings = Array(
            ItemScore(0, 1),
            ItemScore(2, 2)
        )

        val predictions = new SARScalaPredictor(
            itemOffsets,
            itemIds,
            itemValues).predict(25, userRatings)

        for (p <- predictions) {
            // println(s"$p.u1 to $p.i1")
            println(p)
        }
    }
}