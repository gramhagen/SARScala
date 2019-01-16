package gramhagen.sarscala

import scala.collection.mutable.{ArrayBuilder, HashSet, ListBuffer, PriorityQueue}
import util.control.Breaks._
import gramhagen.sarscala.SARExtensions._

class SARScalaPredictor (
    itemOffsets: Array[Long],
    itemIds: Array[Int],
    itemValues: Array[Double] // rename to similarity scores
) {

    def predict(u1:Int, userRatings: Array[ItemScore]): TraversableOnce[UserAffinity] = {
        val topK = 10
        val removeSeen = false

        // TODO handle empty users

        // TODO sort itemsOfUsers + ratings in parallel by itemId
        // -> done in getProcessedRatings

        val seenItems = if(removeSeen) HashSet(userRatings.map(r => r.id): _*) else HashSet[Int]()

        val topKitems = PriorityQueue[ItemScore]()

        // loop through items user has seen
        for (itemOfUser <- userRatings) {
            val iid = itemOfUser.id
            val relatedStart = itemOffsets(iid)
            /*
            if (itemOffsets.length <= (iid + 1)) {
                val iol = itemOffsets.length
                println(s"Unknown iid? : $iid +1 >= $iol")
            }
            */
            val relatedEnd = itemOffsets(iid + 1)

            // println(s"past item $iid ")

            for (relatedItemIdx <- relatedStart until relatedEnd) {
              val relatedItem = itemIds(relatedItemIdx.toInt) // TODO: int to long?
              // println(s"relatedItem $relatedItem (at $relatedItemIdx in relatedStart-$relatedEnd)")

              // avoid duplicate             
              if (!(seenItems contains relatedItem)) {
                seenItems += relatedItem

                // val relatedItemScore
                // TODO: move to method(join_prod_sum)
                var contribStart = itemOffsets(relatedItem).toInt // TODO: int to long?
                val contribEnd = itemOffsets(relatedItem + 1).toInt

                var userIidStart = 0
                val userIidEnd = userRatings.length

                var score = 0.0f

                breakable {
                  while (true) {
                    var userIidItem = userRatings(userIidStart).id
                    var contribItem = itemIds(contribStart)

                    // println(s"join $userIidItem <-> $contribItem")
                    // binary search
                    if (userIidItem < contribItem) {
                      userIidStart = userRatings.searchIndex(userIidStart, userIidEnd, contribItem, (a:ItemScore, b:Int) => a.id.compareTo(b))
                      if (userIidStart == -1)
                        break
                    }
                    else if (userIidItem > contribItem) {
                      contribStart = itemIds.searchIndex(contribStart, contribEnd, userIidItem, (a, b:Int) => a.compareTo(b))
                      if (contribStart == -1)
                        break
                    }
                    else {
                      score += (userRatings(userIidStart).score * itemValues(contribStart)).toFloat

                      userIidStart += 1
                      if (userIidStart == userIidEnd)
                        break

                      contribStart += 1
                      if (contribStart == contribEnd)
                        break
                    }
                  }
                }

                if (score > 0) {
                  if (topKitems.length < topK)
                    topKitems.enqueue(ItemScore(relatedItem, score))
                  else if (topKitems.head.score < score) {
                    topKitems.dequeue
                    topKitems.enqueue(ItemScore(relatedItem, score))
                  }
                }
              }
            }
        }

        topKitems.map(is => UserAffinity(u1, is.id, is.score))
    }
}