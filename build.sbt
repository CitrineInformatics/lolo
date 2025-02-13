import Dependencies._

ThisBuild / name := "lolo"
ThisBuild / scalaVersion := "2.13.10"
ThisBuild / crossScalaVersions := List("2.13.10")
ThisBuild / organization := "io.citrine"
ThisBuild / organizationName := "Citrine Informatics"
ThisBuild / homepage := Some(url("https://github.com/CitrineInformatics/lolo"))
ThisBuild / description := "A random forest-centered machine learning library in Scala."
ThisBuild / licenses += "Apache-2.0" -> url("http://www.apache.org/licenses/LICENSE-2.0.txt")

ThisBuild / developers := List(
  Developer(
    id = "mVenetos97",
    name = "Maxwell C. Venetos",
    email = "mvenetos@citrine.io",
    url = url("https://github.com/mVenetos97")
  )
)
ThisBuild / scmInfo := Some(
  ScmInfo(url("https://github.com/CitrineInformatics/lolo"), "scm:git@github.com:CitrineInformatics/lolo.git")
)
ThisBuild / sonatypeCredentialHost := "s01.oss.sonatype.org"
ThisBuild / sonatypeRepository := "https://s01.oss.sonatype.org/service/local"
ThisBuild / pomIncludeRepository := { _ => false }

ThisBuild / libraryDependencies ++= loloDependencies ++ testDependencies
ThisBuild / Test / testOptions += Tests.Argument(TestFrameworks.JUnit, "-v")

// Assembly settings
ThisBuild / assembly / test := {}
ThisBuild / assembly / assemblyJarName := s"lolo-jar-with-dependencies.jar"

// Allows for creation of a fat jar by specifying merge strategy when dependencies conflict
// Run `sbt assembly` to create fat jar and save in target/scala-X.X.X directory
ThisBuild / assembly / assemblyMergeStrategy := {
  case PathList(ps @ _*) if ps.last endsWith ".class" => MergeStrategy.first
  case x =>
    val oldStrategy = (ThisBuild / assembly / assemblyMergeStrategy).value
    oldStrategy(x)
}

enablePlugins(SiteScaladocPlugin)
enablePlugins(GhpagesPlugin)
ThisBuild / git.remoteRepo := "git@github.com:CitrineInformatics/lolo.git"
