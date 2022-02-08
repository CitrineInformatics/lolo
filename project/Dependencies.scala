import sbt._

object Dependencies {
  lazy val thetaVersion = "1.1.4"
  lazy val breezeVersion = "2.0"
  lazy val junitVersion = "4.13.1"
  lazy val scalaTestVersion = "3.2.2"

  // this is used in build.sbt so that it works with both 2.12 and 2.13
  lazy val scalaParallelCollectionsVersion = "1.0.0"

  val loloDeps = Seq(
    "io.citrine"      %% "theta"           % thetaVersion,
    "org.scalanlp"    %% "breeze"          % breezeVersion,
    "junit"            % "junit"           % junitVersion     % "test",
    "org.scalatest"   %% "scalatest"       % scalaTestVersion % "test",
    "com.novocode"     % "junit-interface" % "0.11"           % "test",
    "org.knowm.xchart" % "xchart"          % "3.5.2"
  )
}
