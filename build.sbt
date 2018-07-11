import Dependencies._

name             := "lolo"
scalaVersion     := "2.12.4"
version          := "1.0.0"
organization     := "io.citrine"
organizationName := "Citrine Informatics"
homepage         := Some(url("https://github.com/CitrineInformatics/lolo"))
developers       := List(Developer(id="maxhutch", name="Max Hutchinson", email="maxhutch@citrine.io", url=url("https://github.com/maxhutch")))
description      := "A random forest-centered machine learning library in Scala."
licenses         += "Apache-2.0" ->  url("http://www.apache.org/licenses/LICENSE-2.0.txt")
scmInfo          := Some(ScmInfo(url("https://github.com/CitrineInformatics/lolo"), "scm:git@github.com:CitrineInformatics/lolo.git"))
credentials      += Credentials("Sonatype Nexus Repository Manager", "oss.sonatype.org", sys.env.get("OSSRH_JIRA_USERNAME").getOrElse(""), sys.env.get("OSSRH_JIRA_PASSWORD").getOrElse(""))

pomIncludeRepository := { _ => false }
publishMavenStyle := true
publishTo := {
  val nexus = "https://oss.sonatype.org/"
  if (isSnapshot.value)
    Some("snapshots" at nexus + "content/repositories/snapshots")
  else
    Some("releases"  at nexus + "service/local/staging/deploy/maven2")
}

useGpg := true
pgpPassphrase := Some(sys.env.get("GPG_PASSPHRASE").getOrElse("").toArray)
usePgpKeyHex(sys.env.get("GPG_KEY_NAME").getOrElse("0"))


libraryDependencies ++= loloDeps

enablePlugins(SiteScaladocPlugin)
enablePlugins(GhpagesPlugin)
git.remoteRepo := "git@github.com:CitrineInformatics/lolo.git"
