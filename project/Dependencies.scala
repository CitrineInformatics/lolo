import sbt._

object Dependencies {

  object Versions {
    val theta = "1.1.5"
    val sprandom = "0.1.1"
    val breeze = "2.1.0"
    val xchart = "3.8.2"

    val slf4j = "2.0.3"
    val junit = "4.13.2"
    val scalaTest = "3.2.14"
    val log4j = "2.19.0"
  }

  lazy val loloDependencies = Seq(
    "org.scala-lang.modules" %% "scala-parallel-collections" % "1.0.4",
    "io.citrine"             %% "theta"                      % Versions.theta,
    "io.citrine"             %% "sprandom"                   % Versions.sprandom,
    "org.scalanlp"           %% "breeze"                     % Versions.breeze,
    "org.knowm.xchart"        % "xchart"                     % Versions.xchart,
    "org.slf4j"               % "slf4j-api"                  % Versions.slf4j
  )

  lazy val testDependencies = Seq(
    "com.github.sbt"           % "junit-interface"   % "0.13.3",
    "junit"                    % "junit"             % Versions.junit,
    "org.scalatest"           %% "scalatest"         % Versions.scalaTest,
    "org.apache.logging.log4j" % "log4j-core"        % Versions.log4j,
    "org.apache.logging.log4j" % "log4j-api"         % Versions.log4j,
    "org.apache.logging.log4j" % "log4j-slf4j2-impl" % Versions.log4j
  ).map(_ % Test)
}
