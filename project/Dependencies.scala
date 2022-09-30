import sbt._

object Dependencies {
  lazy val thetaVersion = "1.1.5"
  lazy val sprandomVersion = "0.1.1"
  lazy val breezeVersion = "2.0"
  lazy val junitVersion = "4.13.1"
  lazy val scalaTestVersion = "3.2.2"
  lazy val scalaParallelCollectionsVersion = "1.0.0"

  lazy val logging = Seq(
    "org.slf4j"                % "slf4j-api"        % "1.7.36",
    "org.apache.logging.log4j" % "log4j-core"       % "2.17.2",
    "org.apache.logging.log4j" % "log4j-api"        % "2.17.2",
    "org.apache.logging.log4j" % "log4j-slf4j-impl" % "2.17.2"
  )

  lazy val loloDeps = Seq(
    "io.citrine"              %% "theta"                      % thetaVersion,
    "io.citrine"              %% "sprandom"                   % sprandomVersion,
    "org.scalanlp"            %% "breeze"                     % breezeVersion,
    "org.scala-lang.modules"  %% "scala-parallel-collections" % scalaParallelCollectionsVersion,
    "junit"                    % "junit"                      % junitVersion     % "test",
    "org.scalatest"           %% "scalatest"                  % scalaTestVersion % "test",
    "com.github.sbt"           % "junit-interface"            % "0.13.3"         % "test",
    "org.knowm.xchart"         % "xchart"                     % "3.5.2"
  )
}
