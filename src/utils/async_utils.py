"""
Async utilities for managing concurrent operations in the RAG system.
"""
import asyncio
import functools
import time
from typing import Any, Callable, Coroutine, List, TypeVar, Union, Optional
from concurrent.futures import ThreadPoolExecutor
from .logger import get_logger

logger = get_logger(__name__)

T = TypeVar('T')

def async_timed(func: Callable) -> Callable:
    """
    Decorator to time async functions and log execution time.
    
    Example:
        @async_timed
        async def my_function():
            await asyncio.sleep(1)
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            elapsed = time.time() - start
            logger.debug(f"{func.__name__} took {elapsed:.2f} seconds")
    return wrapper

async def gather_with_concurrency(
    n: int,
    *coroutines: Coroutine[Any, Any, T],
    return_exceptions: bool = False
) -> List[Union[T, Exception]]:
    """
    Gather coroutines with a concurrency limit.
    
    Args:
        n: Maximum number of concurrent coroutines
        *coroutines: Coroutines to execute
        return_exceptions: Whether to return exceptions or raise them
    
    Returns:
        List of results in the same order as coroutines
        
    Example:
        results = await gather_with_concurrency(
            5,  # Max 5 concurrent tasks
            task1(),
            task2(),
            task3()
        )
    """
    semaphore = asyncio.Semaphore(n)
    
    async def wrapped_coro(coro: Coroutine) -> T:
        async with semaphore:
            return await coro
    
    wrapped_coros = [wrapped_coro(c) for c in coroutines]
    return await asyncio.gather(*wrapped_coros, return_exceptions=return_exceptions)

def run_async_in_thread(async_func: Callable[..., Coroutine]) -> Callable:
    """
    Run an async function in a separate thread with its own event loop.
    Useful for integrating async code with synchronous code.
    
    Example:
        @run_async_in_thread
        async def fetch_data():
            return await some_async_function()
        
        # Can be called from sync code
        result = fetch_data()
    """
    @functools.wraps(async_func)
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(async_func(*args, **kwargs))
        finally:
            loop.close()
    return wrapper

class AsyncTaskManager:
    """
    Manage multiple async tasks with lifecycle management.
    Useful for long-running operations.
    
    Example:
        manager = AsyncTaskManager(max_concurrent=10)
        
        # Start tasks
        task1 = await manager.run_task(fetch_data1())
        task2 = await manager.run_task(fetch_data2())
        
        # Wait for all to complete
        await manager.wait_all()
        
        # Clean up
        await manager.close()
    """
    
    def __init__(self, max_concurrent: int = 10):
        self.max_concurrent = max_concurrent
        self.tasks: List[asyncio.Task] = []
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._closed = False
    
    async def run_task(self, coro: Coroutine) -> asyncio.Task:
        """
        Run a task with concurrency control.
        """
        if self._closed:
            raise RuntimeError("TaskManager is closed")
        
        async with self._semaphore:
            task = asyncio.create_task(coro)
            self.tasks.append(task)
            task.add_done_callback(lambda t: self.tasks.remove(t))
            return task
    
    async def wait_all(self, timeout: Optional[float] = None):
        """
        Wait for all tasks to complete.
        """
        if self.tasks:
            await asyncio.wait(self.tasks, timeout=timeout)
    
    async def cancel_all(self):
        """
        Cancel all running tasks.
        """
        for task in self.tasks:
            if not task.done():
                task.cancel()
        
        if self.tasks:
            await asyncio.wait(self.tasks)
    
    async def close(self):
        """
        Clean up all tasks.
        """
        self._closed = True
        await self.cancel_all()

async def run_parallel(*coroutines: Coroutine) -> List[Any]:
    """
    Simple helper to run coroutines in parallel.
    
    Example:
        results = await run_parallel(
            process_file1(),
            process_file2(),
            process_file3()
        )
    """
    return await asyncio.gather(*coroutines)

def create_task(coro: Coroutine) -> asyncio.Task:
    """
    Create a task with proper error logging.
    
    Example:
        task = create_task(long_running_operation())
    """
    task = asyncio.create_task(coro)
    
    def log_error(task):
        if not task.cancelled() and task.exception():
            logger.error(f"Task failed: {task.exception()}")
    
    task.add_done_callback(log_error)
    return task