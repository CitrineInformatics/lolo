import Dependencies._

name := "lolo"
scalaVersion := "2.13.4"
crossScalaVersions := List("2.13.4", "2.12.4")
organization := "io.citrine"
organizationName := "Citrine Informatics"
homepage := Some(url("https://github.com/CitrineInformatics/lolo"))
developers := List(
  Developer(
    id = "maxhutch",
    name = "Max Hutchinson",
    email = "maxhutch@citrine.io",
    url = url("https://github.com/maxhutch")
  )
)
description := "A random forest-centered machine learning library in Scala."
licenses += "Apache-2.0" -> url("http://www.apache.org/licenses/LICENSE-2.0.txt")
scmInfo := Some(
  ScmInfo(url("https://github.com/CitrineInformatics/lolo"), "scm:git@github.com:CitrineInformatics/lolo.git")
)
sonatypeCredentialHost := "s01.oss.sonatype.org"
sonatypeRepository := "https://s01.oss.sonatype.org/service/local"

pomIncludeRepository := { _ => false }
test in assembly := {}
assemblyJarName in assembly := s"lolo-jar-with-dependencies.jar"

libraryDependencies ++= loloDeps
libraryDependencies ++= {
  CrossVersion.partialVersion(scalaVersion.value) match {
    case Some((2, major)) if major <= 12 =>
      Seq()
    case _ =>
      Seq("org.scala-lang.modules" %% "scala-parallel-collections" % scalaParallelCollectionsVersion)
  }
}

// Allows for creation of a fat jar by specifying merge strategy when dependencies conflict
// Run `sbt assembly` to create fat jar and save in target/scala-X.X.X directory
assemblyMergeStrategy in assembly := {
  case PathList(ps @ _*) if ps.last endsWith ".class" => MergeStrategy.first
  case x: Any =>
    val oldStrategy = (assemblyMergeStrategy in assembly).value
    oldStrategy(x)
}

enablePlugins(SiteScaladocPlugin)
enablePlugins(GhpagesPlugin)
git.remoteRepo := "git@github.com:CitrineInformatics/lolo.git"
