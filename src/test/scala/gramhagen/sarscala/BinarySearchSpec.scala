package gramhagen.sarscala

import gramhagen.sarscala.SARExtensions._
import org.scalatest._

class BinarySearchSpec extends FlatSpec {

    "binarySearch" should "return index" in  {
        assert(0 == Array(1,2,3).searchIndex(1, (a, b:Int) => a.compare(b)))
        assert(1 == Array(1,2,3).searchIndex(2, (a, b:Int) => a.compare(b)))
        assert(2 == Array(1,2,3).searchIndex(3, (a, b:Int) => a.compare(b)))
    }

    "binarySearch" should "return -1" in  {
        assert(-1 == Array(1,2,3).searchIndex(0, (a, b:Int) => a.compare(b)))
        assert(-1 == Array(1,2,3).searchIndex(4, (a, b:Int) => a.compare(b)))
        assert(-1 == Array(1,2,4).searchIndex(3, (a, b:Int) => a.compare(b)))
    }

    "binarySearch for empty" should "return -1" in {
        assert(-1 == Array[Int]().searchIndex(0, (a, b:Int) => a.compare(b)))
    }
}