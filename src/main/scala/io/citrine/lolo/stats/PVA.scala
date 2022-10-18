package io.citrine.lolo.stats

case class PVA[+T](inputs: Vector[Any], predicted: T, actual: T)
