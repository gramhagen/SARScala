package gramhagen.sarscala

import scala.annotation.tailrec

object SARExtensions {
    // <% Ordered[T] 
    implicit class BinarySearch[T] (arr:Array[T]) {
        def searchIndex[S](value:S, comparator:(T, S) => Int): Int =
            searchIndex(0, arr.length, value, comparator)

        @tailrec
        final def searchIndex[S](start:Int, end:Int, value:S, comparator:(T, S) => Int): Int = {
            if (end <= start)
                return -1

            val m = start + ((end - start) / 2)

            comparator(arr(m), value) match {
                case c if c < 0 => searchIndex(m + 1, end, value, comparator)
                case c if c > 0 => searchIndex(start, m, value, comparator)
                case _ => m
            }
        }
    }
}