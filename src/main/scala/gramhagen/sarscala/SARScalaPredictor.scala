package gramhagen.sarscala

import scala.collection.mutable.{ArrayBuilder, HashSet, ListBuffer, PriorityQueue}
import util.control.Breaks._
import gramhagen.sarscala.SARExtensions._

class SARScalaPredictor[T] (
    itemOffsets: Array[Int],
    itemIds: Array[Int],
    itemValues: Array[Float] // rename to similarity scores
) {

    def predict(u1:T, userRatings:Array[ItemScore], topK:Int): TraversableOnce[UserAffinity[T]] = {
        val removeSeen = true

        // TODO handle empty users

        // userRatings must be sorted -> done in getProcessedRatings
        val seenItems = if(removeSeen) HashSet(userRatings.map(r => r.id): _*) else HashSet[Int]()

        val topKitems = PriorityQueue[ItemScore]()

        // loop through items user has seen
        for (itemOfUser <- userRatings) {
          val iid = itemOfUser.id
          val relatedStart = itemOffsets(iid)
          val relatedEnd = itemOffsets(iid + 1)

          for (relatedItemIdx <- relatedStart until relatedEnd) {
            val relatedItem = itemIds(relatedItemIdx)

            // avoid duplicate             
            if (!(seenItems contains relatedItem)) {
              seenItems += relatedItem

              val score = joinProdSum(itemOffsets, userRatings, relatedItem)

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

    def joinProdSum(itemOffsets:Array[Int], userRatings:Array[ItemScore], relatedItem:Int): Float = {
      var contribStart = itemOffsets(relatedItem).toInt
      val contribEnd = itemOffsets(relatedItem + 1).toInt

      var userIidStart = 0
      val userIidEnd = userRatings.length
      var score = 0.0f

      while (true) {
        var userIidItem = userRatings(userIidStart).id
        var contribItem = itemIds(contribStart)

        // binary search
        if (userIidItem < contribItem) {
          userIidStart = userRatings.lowerBound(userIidStart, userIidEnd, contribItem, -1, (a:ItemScore, b:Int) => a.id.compareTo(b))
          if (userIidStart == userIidEnd)
            return score
        }
        else if (userIidItem > contribItem) {
          contribStart = itemIds.lowerBound(contribStart, contribEnd, userIidItem, -1, (a, b:Int) => a.compareTo(b))
          if (contribStart == contribEnd)
            return score
        }
        else {
          val s1 = userRatings(userIidStart).score
          val s2 = itemValues(contribStart)

          score += (userRatings(userIidStart).score * itemValues(contribStart))

          userIidStart += 1
          if (userIidStart == userIidEnd)
            return score

          contribStart += 1
          if (contribStart == contribEnd)
            return score
        }
      }

      return score
    }
}