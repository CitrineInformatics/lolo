import sbt._

object Dependencies {
  lazy val netlibVersion = "1.1.2"
  lazy val junitVersion = "4.12"
  lazy val thetaVersion = "1.0.0"
  lazy val breezeVersion = "0.13.2"
  lazy val scalaticVersion = "3.0.5"

  val loloDeps = Seq(
    "com.github.fommil.netlib" % "all" % netlibVersion,
    "junit" % "junit" % junitVersion % "test",
    "io.citrine" % "theta" % thetaVersion,
    "org.scalactic" %% "scalactic" % scalaticVersion,
    "org.scalatest" %% "scalatest" % scalaticVersion % "test",
    "com.novocode" % "junit-interface" % "0.11" % "test"
  )
}
