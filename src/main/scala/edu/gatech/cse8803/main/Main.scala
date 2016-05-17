/**
 * @author Hengte Lin
 */

package edu.gatech.cse8803.main

import java.text.SimpleDateFormat
import java.util.Date

import edu.gatech.cse8803.clustering.{NMF, Metrics}
import edu.gatech.cse8803.features.FeatureConstruction
import edu.gatech.cse8803.ioutils.CSVUtils
import edu.gatech.cse8803.model.{Diagnostic, LabResult, Medication}
import edu.gatech.cse8803.phenotyping.T2dmPhenotype
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.clustering.{GaussianMixture, KMeans}
import org.apache.spark.mllib.linalg.{DenseMatrix, Matrices, Vectors, Vector}
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}

import scala.io.Source


object Main {
  def main(args: Array[String]) {
    import org.apache.log4j.Logger
    import org.apache.log4j.Level

    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    val sc = createContext
    val sqlContext = new SQLContext(sc)

    /** initialize loading of data */
    val (medication, labResult, diagnostic) = loadRddRawData(sqlContext)
    val (candidateMedication, candidateLab, candidateDiagnostic) = loadLocalRawData

    /** conduct phenotyping */
    val phenotypeLabel = T2dmPhenotype.transform(medication, labResult, diagnostic)

    /** feature construction with all features */
    val featureTuples = sc.union(
      FeatureConstruction.constructDiagnosticFeatureTuple(diagnostic),
      FeatureConstruction.constructLabFeatureTuple(labResult),
      FeatureConstruction.constructMedicationFeatureTuple(medication)
    )

    val rawFeatures = FeatureConstruction.construct(sc, featureTuples)

    val (kMeansPurity, gaussianMixturePurity, nmfPurity) = testClustering(phenotypeLabel, rawFeatures)
    println(f"[All feature] purity of kMeans is: $kMeansPurity%.5f")
    println(f"[All feature] purity of GMM is: $gaussianMixturePurity%.5f")
    println(f"[All feature] purity of NMF is: $nmfPurity%.5f")


    /** feature construction with filtered features */
    val filteredFeatureTuples = sc.union(
      FeatureConstruction.constructDiagnosticFeatureTuple(diagnostic, candidateDiagnostic),
      FeatureConstruction.constructLabFeatureTuple(labResult, candidateLab),
      FeatureConstruction.constructMedicationFeatureTuple(medication, candidateMedication)
    )

    val filteredRawFeatures = FeatureConstruction.construct(sc, filteredFeatureTuples)

    val (kMeansPurity2, gaussianMixturePurity2, nmfPurity2) = testClustering(phenotypeLabel, filteredRawFeatures)
    println(f"[Filtered feature] purity of kMeans is: $kMeansPurity2%.5f")
    println(f"[Filtered feature] purity of GMM is: $gaussianMixturePurity2%.5f")
    println(f"[Filtered feature] purity of NMF is: $nmfPurity2%.5f")
  }

  def testClustering(phenotypeLabel: RDD[(String, Int)], rawFeatures:RDD[(String, Vector)]): (Double, Double, Double) = {
    import org.apache.spark.mllib.linalg.Matrix
    import org.apache.spark.mllib.linalg.distributed.RowMatrix

    /** scale features */
    val scaler = new StandardScaler(withMean = true, withStd = true).fit(rawFeatures.map(_._2))
    val features = rawFeatures.map({ case (patientID, featureVector) => (patientID, scaler.transform(Vectors.dense(featureVector.toArray)))})
    val rawFeatureVectors = features.map(_._2).cache()
    val rawFeatureIDs=features.map(_._1)

    /** reduce dimension */
    val mat: RowMatrix = new RowMatrix(rawFeatureVectors)
    val pc: Matrix = mat.computePrincipalComponents(10) // Principal components are stored in a local dense matrix.
    val featureVectors = mat.multiply(pc).rows

    val densePc = Matrices.dense(pc.numRows, pc.numCols, pc.toArray).asInstanceOf[DenseMatrix]
    def transform(feature: Vector): Vector = {
      val scaled = scaler.transform(Vectors.dense(feature.toArray))
      Vectors.dense(Matrices.dense(1, scaled.size, scaled.toArray).multiply(densePc).toArray)
    }

    /**  K Means Clustering using spark mllib
      *  Train a k means model using the variabe featureVectors as input
      *  Set maxIterations =20 and seed as 0L
      *  Assign each feature vector to a cluster(predicted Class)
      **/
    val numClusters=3
    val  numIterations=20
    featureVectors.cache()
    val kMeanCluster=KMeans.train(featureVectors, numClusters, numIterations,1,"k-means||",0L).predict(featureVectors)
    val kmeanresult=rawFeatureIDs.zip(kMeanCluster)
    val compareKMean=kmeanresult.join(phenotypeLabel).map(_._2)
    val kMeansPurity = Metrics.purity(compareKMean)



    /** TODO: GMMM Clustering using spark mllib
      *  Train a Gaussian Mixture model using the variabe featureVectors as input
      *  Set maxIterations =20 and seed as 0L
      *  Assign each feature vector to a cluster(predicted Class)
      **/
    val GMMClusters= new GaussianMixture().setK(numClusters).setMaxIterations(numIterations).setSeed(0L).run(featureVectors).predict(featureVectors)
    val GMMResult=rawFeatureIDs.zip(GMMClusters)
    val compareGMM=GMMResult.join(phenotypeLabel).map(_._2)
    val gaussianMixturePurity = Metrics.purity(compareGMM)



    /** NMF */
    val (w, _) = NMF.run(new RowMatrix(rawFeatureVectors), 3, 200)
    // for each row (patient) in W matrix, the index with the max value should be assigned as its cluster type
    val assignments = w.rows.map(_.toArray.zipWithIndex.maxBy(_._1)._2)

    val labels = features.join(phenotypeLabel).map({ case (patientID, (feature, realClass)) => realClass})

    // zip assignment and label into a tuple for computing purity 
    val nmfClusterAssignmentAndLabel = assignments.zipWithIndex().map(_.swap).join(labels.zipWithIndex().map(_.swap)).map(_._2)
    val nmfPurity = Metrics.purity(nmfClusterAssignmentAndLabel)

    (kMeansPurity, gaussianMixturePurity, nmfPurity)
  }

  /**
   * load the sets of string for filtering of medication
   * lab result and diagnostics
    *
    * @return
   */
  def loadLocalRawData: (Set[String], Set[String], Set[String]) = {
    val candidateMedication = Source.fromFile("data/med_filter.txt").getLines().map(_.toLowerCase).toSet[String]
    val candidateLab = Source.fromFile("data/lab_filter.txt").getLines().map(_.toLowerCase).toSet[String]
    val candidateDiagnostic = Source.fromFile("data/icd9_filter.txt").getLines().map(_.toLowerCase).toSet[String]
    (candidateMedication, candidateLab, candidateDiagnostic)
  }

  def loadRddRawData(sqlContext: SQLContext): (RDD[Medication], RDD[LabResult], RDD[Diagnostic]) = {

    val dateFormat = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ssX")
    val medicationdata=CSVUtils.loadCSVAsTable(sqlContext,"data/medication_orders_INPUT.csv","MedicationTable")
    val labResultdata=CSVUtils.loadCSVAsTable(sqlContext,"data/lab_results_INPUT.csv","LabTable")
    val IDdata=CSVUtils.loadCSVAsTable(sqlContext,"data/encounter_INPUT.csv","IDTable")
    val diagnosedata=CSVUtils.loadCSVAsTable(sqlContext,"data/encounter_dx_INPUT.csv","DiagTable")

    /** load data using Spark SQL into three RDDs and return them
      * */


    val RDDrowmed= sqlContext.sql("SELECT Member_ID AS patientID, Order_Date AS date, Drug_Name AS medicine  FROM MedicationTable")
    val RDDrowdiag= sqlContext.sql("SELECT IDTable.Member_ID AS patientID, IDTable.Encounter_DateTime AS date, DiagTable.code AS code  FROM IDTable INNER JOIN DiagTable ON IDTable.Encounter_ID= DiagTable.Encounter_ID")
    val RDDrowlab=  sqlContext.sql("SELECT Member_ID AS patientID, Date_Resulted AS date, Result_Name AS testName, Numeric_Result as value  FROM LabTable WHERE Numeric_Result!=''")
    val medication: RDD[Medication] = RDDrowmed.map(p => Medication(p(0).asInstanceOf[String],dateFormat.parse(p(1).asInstanceOf[String]),p(2).asInstanceOf[String].toLowerCase))
    val labResult: RDD[LabResult] = RDDrowlab.map(p => LabResult(p(0).asInstanceOf[String],dateFormat.parse(p(1).asInstanceOf[String]),p(2).asInstanceOf[String].toLowerCase,p(3).asInstanceOf[String].filterNot(",".toSet).toDouble))
    val diagnostic: RDD[Diagnostic] =  RDDrowdiag.map(p => Diagnostic(p(0).asInstanceOf[String],dateFormat.parse(p(1).asInstanceOf[String]),p(2).asInstanceOf[String].toLowerCase))

    (medication, labResult, diagnostic)
  }

  def createContext(appName: String, masterUrl: String): SparkContext = {
    val conf = new SparkConf().setAppName(appName).setMaster(masterUrl)
    new SparkContext(conf)
  }

  def createContext(appName: String): SparkContext = createContext(appName, "local")

  def createContext: SparkContext = createContext("CSE 8803 Homework Two Application", "local")

  //----------------------
  def percentage(clusterAssignmentAndLabel: RDD[(Int, Int)]): List[((Int,Int),Int)] ={
    val result=clusterAssignmentAndLabel.map(f =>((f._1,f._2),1)).keyBy(_._1).reduceByKey((x,y) => (x._1,x._2+y._2)).map(f => f._2).collect.toList

    result
  }
}
