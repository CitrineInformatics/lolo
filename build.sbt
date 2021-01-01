import Dependencies._

name             := "lolo"
scalaVersion     := "2.12.4"
organization     := "io.citrine"
organizationName := "Citrine Informatics"
homepage         := Some(url("https://github.com/CitrineInformatics/lolo"))
developers       := List(Developer(id="maxhutch", name="Max Hutchinson", email="maxhutch@citrine.io", url=url("https://github.com/maxhutch")))
description      := "A random forest-centered machine learning library in Scala."
licenses         += "Apache-2.0" ->  url("http://www.apache.org/licenses/LICENSE-2.0.txt")
scmInfo          := Some(ScmInfo(url("https://github.com/CitrineInformatics/lolo"), "scm:git@github.com:CitrineInformatics/lolo.git"))

pomIncludeRepository := { _ => false }

libraryDependencies ++= loloDeps

enablePlugins(SiteScaladocPlugin)
enablePlugins(GhpagesPlugin)
git.remoteRepo := "git@github.com:CitrineInformatics/lolo.git"
