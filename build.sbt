import Dependencies._

lazy val root = (project in file("."))
  .settings(
    name             := "lolo",
    scalaVersion     := "2.12.4",
    version          := "1.0.0",
    organization     := "io.citrine",
    organizationName := "Citrine Informatics",
    homepage         := Some(url("https://github.com/CitrineInformatics/lolo")),
    description      := "A random forest-centered machine learning library in Scala.",
    licenses         += "Apache2" ->  url("http://www.apache.org/licenses/LICENSE-2.0.txt"),
    libraryDependencies ++= loloDeps
  )
