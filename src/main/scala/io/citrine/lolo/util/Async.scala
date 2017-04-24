package io.citrine.lolo.util

/**
  * Object containing utility functions for supporting interrupts
  * Created by maxhutch on 4/20/17.
  */
object Async {
  /**
    * Check the thread's interrupted status, because it could stop
    */
  def canStop(): Unit = {
    if (Thread.interrupted()) throw new InterruptedException()
  }
}