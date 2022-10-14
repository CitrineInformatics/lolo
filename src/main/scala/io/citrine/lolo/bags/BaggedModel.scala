package io.citrine.lolo.bags

import io.citrine.lolo.Model

trait BaggedModel[+T] extends Model[T] {

  override def transform(inputs: Seq[Vector[Any]]): BaggedResult[T]
}
