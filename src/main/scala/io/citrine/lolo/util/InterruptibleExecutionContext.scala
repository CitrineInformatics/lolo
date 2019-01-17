package io.citrine.lolo.util

import java.util.concurrent.Executors

import scala.concurrent.ExecutionContext

/**
  * Checks if thread was interrupted before performing a task.
  *
  * Thin wrapper around the global execution context
  * Created by maxhutch on 4/20/17.
  */
class InterruptibleExecutionContext(executionContext: ExecutionContext) extends ExecutionContext {
  override def execute(runnable: Runnable): Unit = {
    Async.canStop()
    executionContext.execute(runnable)
  }

  override def reportFailure(cause: Throwable): Unit = {
    executionContext.reportFailure(cause)
  }
}

/**
  * Provide default InterruptibleExecutionContext based on the global EC
  */
object InterruptibleExecutionContext {
  private val default = new InterruptibleExecutionContext(
    ExecutionContext.global
  )

  def apply(): InterruptibleExecutionContext = default
}
