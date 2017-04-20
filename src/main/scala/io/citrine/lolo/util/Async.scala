package io.citrine.lolo.util

/**
  * Created by maxhutch on 4/20/17.
  */
object Async {
  /**
    * Check the thread's interuppted status, because it could stop
    */
  def canStop(): Unit = {
    if (Thread.interrupted()) throw new InterruptedException()
  }
}