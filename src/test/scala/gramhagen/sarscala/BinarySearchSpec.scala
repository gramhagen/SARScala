package gramhagen.sarscala

import gramhagen.sarscala.SARExtensions._
import org.scalatest._

class BinarySearchSpec extends FlatSpec {

    "binarySearch" should "return index" in  {
        assert(0 == Array(1,2,3).lowerBound(1, (a, b:Int) => a.compare(b)))
        assert(1 == Array(1,2,3).lowerBound(2, (a, b:Int) => a.compare(b)))
        assert(2 == Array(1,2,3).lowerBound(3, (a, b:Int) => a.compare(b)))
    }

    "binarySearch for empty" should "return -1" in {
        assert(-1 == Array[Int]().lowerBound(0, (a, b:Int) => a.compare(b)))
    }

    "binarySearch for lower" should "return next lower" in {
        assert(0 == Array(1,3,5).lowerBound(0, (a, b:Int) => a.compare(b)))
        assert(1 == Array(1,3,5).lowerBound(2, (a, b:Int) => a.compare(b)))
        assert(2 == Array(1,3,5).lowerBound(4, (a, b:Int) => a.compare(b)))
        assert(3 == Array(1,3,5).lowerBound(6, (a, b:Int) => a.compare(b)))
    }

    "binarySearch for lower large" should "return next lower" in {
        // println(arr.mkString(","))
        val arrIdx = Range(0,100)
        val arr = arrIdx.map({ _*2 }).toArray

        for (i <- arrIdx) {
            assert(i == arr.lowerBound(2*i-1, (a, b:Int) => a.compare(b)))
            assert(i == arr.lowerBound(2*i, (a, b:Int) => a.compare(b)))
            assert(i+1 == arr.lowerBound(2*i+1, (a, b:Int) => a.compare(b)))
        }  
    }
}