package io.citrine.lolo.util

import scala.concurrent.ExecutionContext

/**
  * Checks if thread was interrupted before performing a task.
  *
  * Thin wrapper around the global execution context
  * Created by maxhutch on 4/20/17.
  */
object InterruptibleExecutionContext extends ExecutionContext {
  override def execute(runnable: Runnable): Unit = {
    if (Thread.interrupted()) throw new InterruptedException
    ExecutionContext.global.execute(runnable)
  }

  override def reportFailure(cause: Throwable): Unit = {
    ExecutionContext.global.reportFailure(cause)
  }
}
