import sbt._

object Dependencies {
  lazy val netlibVersion = "1.1.2"
  lazy val junitVersion = "4.13.1"
  lazy val thetaVersion = "1.1.2"
  lazy val breezeVersion = "1.1"
  lazy val scalaticVersion = "3.2.2"

  // this is used in build.sbt so that it works with both 2.12 and 2.13
  lazy val scalaParallelCollectionsVersion = "1.0.0"

  val loloDeps = Seq(
    "com.github.fommil.netlib" % "all" % netlibVersion,
    "junit" % "junit" % junitVersion % "test",
    "org.scalanlp" %% "breeze" % breezeVersion,
    "io.citrine" %% "theta" % thetaVersion,
    "org.scalactic" %% "scalactic" % scalaticVersion,
    "org.scalatest" %% "scalatest" % scalaticVersion % "test",
    "com.novocode" % "junit-interface" % "0.11" % "test",
    "org.knowm.xchart" % "xchart" % "3.5.2"
  )
}
