package gramhagen.sarscala

import scala.annotation.tailrec

object SARExtensions {
    implicit class BinarySearch[T] (arr:Array[T]) {
        def lowerBound[S](value:S, comparator:(T, S) => Int): Int =
            lowerBound(0, arr.length, value, -1, comparator)

        // see C++ STL::lower_bound
        @tailrec
        final def lowerBound[S](start:Int, end:Int, value:S, lastIndex:Int, comparator:(T, S) => Int): Int = {
            if (end <= start)
                return lastIndex

            val m = start + ((end - start) / 2)

            comparator(arr(m), value) match {
                case c if c < 0 => searchIndex(m + 1, end, value, end, comparator)
                case c if c > 0 => searchIndex(start, m, value, m, comparator)
                case _ => m
            }
        }
    }
}