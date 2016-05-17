package edu.gatech.cse8803.clustering

/**
  * @author Hengte Lin
  */


import breeze.linalg.{DenseVector => BDV, DenseMatrix => BDM, sum}
import breeze.linalg._
import breeze.numerics._
import org.apache.spark.mllib.linalg.{Matrices, Vectors, Vector}
import org.apache.spark.mllib.linalg.distributed.RowMatrix


object NMF {

  /**
   * Run NMF clustering 
   * @param V The original non-negative matrix 
   * @param k The number of clusters to be formed, also the number of cols in W and number of rows in H
   * @param maxIterations The maximum number of iterations to perform
   * @param convergenceTol The maximum change in error at which convergence occurs.
   * @return two matrixes W and H in RowMatrix and DenseMatrix format respectively 
   */
  def run(V: RowMatrix, k: Int, maxIterations: Int, convergenceTol: Double = 1e-4): (RowMatrix, BDM[Double]) = {

    /**
     *
     * Initialize W, H randomly 
     * Calculate the initial error (Euclidean distance between V and W * H)
     */
    var W = new RowMatrix(V.rows.map(_ => BDV.rand[Double](k)).map(fromBreeze).cache)
    var H = BDM.rand[Double](k, V.numCols().toInt)
    var errorPrevious=0.0
    var errorNow=errorfunction(V,multiply(W,H))
    val sc=W.rows.sparkContext


    /**
     *
     * Iteratively update W, H in a parallel fashion until error falls below the tolerance value 
     * The updating equations are, 
     * H = H.* W^T^V ./ (W^T^W H)
     * W = W.* VH^T^ ./ (W H H^T^)
     */

    while(abs((errorNow-errorPrevious)/errorNow) > convergenceTol){
      W.rows.cache()
      //W.rows.checkpoint()
      V.rows.cache()
      //V.rows.checkpoint()

      val Ws=computeWTV(W,W)

      sc.broadcast(Ws)

      H=(H :* computeWTV(W,V)) :/ ((Ws * H)+ 0.000001)
      val Hs= (H * H.t)
      sc.broadcast(Hs)
      W=dotDiv(dotProd(W,multiply(V,H.t)),multiply(W,Hs))
      errorPrevious=errorNow
      errorNow=errorfunction(V,multiply(W,H))
      W.rows.unpersist(false)
      V.rows.unpersist(false)

      println(errorNow-errorPrevious)


    }



    (W,H)
  }


  /**  
  *
  * Below are recommended helper functions for matrix manipulation
  * For the implementation of the first three helper functions (with a null return), 
  * you can refer to dotProd and dotDiv whose implementation are provided
  */



  /** compute the mutiplication of a RowMatrix and a dense matrix */
  private def errorfunction(V: RowMatrix,WH: RowMatrix): Double = {
    val diff=V.rows.zip(WH.rows).map(f => toBreezeVector(f._1) :- toBreezeVector(f._2)).map(f => f :* f).map(f => sum(f)).reduce((x,y) => x+y)/2
    diff
  }
  private def multiply(X: RowMatrix, d: BDM[Double]): RowMatrix = {
    val rows=X.multiply(fromBreeze(d))

    rows
  }

 /** get the dense matrix representation for a RowMatrix */
  private def getDenseMatrix(X: RowMatrix): BDM[Double] = {
    null
  }

  /** matrix multiplication of W.t and V */
  def computeWTV(W: RowMatrix, V: RowMatrix): BDM[Double] = {
    //val Wrows=W.rows.zip(V.rows).map(f =>( Matrices.dense(f._1.size,1,f._1.toArray).multiply(f._2)))
    val x= W.rows.zip(V.rows).map{
      f =>
      val Wrm= new BDM[Double](f._1.size,1,f._1.toArray)
      val Vrm =  new BDM[Double](1,f._2.size,f._2.toArray)
      val result: BDM[Double] = Wrm * Vrm
      result
    }
    x.reduce(_+_)
  }

  /** dot product of two RowMatrixes */
  def dotProd(X: RowMatrix, Y: RowMatrix): RowMatrix = {
    val rows = X.rows.zip(Y.rows).map{case (v1: Vector, v2: Vector) =>
      toBreezeVector(v1) :* toBreezeVector(v2)
    }.map(fromBreeze)
    new RowMatrix(rows)
  }

  /** dot division of two RowMatrixes */
  def dotDiv(X: RowMatrix, Y: RowMatrix): RowMatrix = {
    val rows = X.rows.zip(Y.rows).map{case (v1: Vector, v2: Vector) =>
      toBreezeVector(v1) :/ toBreezeVector(v2).mapValues(_ + 2.0e-15)
    }.map(fromBreeze)
    new RowMatrix(rows)
  }
}