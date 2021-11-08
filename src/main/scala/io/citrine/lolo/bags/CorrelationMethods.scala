package io.citrine.lolo.bags

/** Enumerates the ways in which uncertainty correlation can be calculated. */
object CorrelationMethods extends Enumeration {
  type CorrelationMethod = Value

  val Trivial, FromTraining, Bootstrap, Jackknife, JackknifeExplicit = Value
}
