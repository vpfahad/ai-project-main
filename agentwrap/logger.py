import functools
import logging
import time
from datetime import datetime
from typing import Optional, Callable, Any
import traceback

# Configure default logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def function_logger(
    level: int = logging.INFO,
    logger_name: Optional[str] = None,
    include_args: bool = True,
    include_return: bool = True,
    include_traceback: bool = True
) -> Callable:
    """
    A decorator that logs function execution details including name, status, and timing.
    
    Args:
        level: Logging level (default: logging.INFO)
        logger_name: Custom logger name (default: uses function module name)
        include_args: Whether to log function arguments (default: True)
        include_return: Whether to log return value (default: True)
        include_traceback: Whether to log full traceback on errors (default: True)
    
    Returns:
        Decorated function with logging capabilities
    
    Example:
        @function_logger()
        def my_function(x, y):
            return x + y
        
        @function_logger(level=logging.DEBUG, include_args=False)
        def debug_function():
            return "debug info"
    """
    def decorator(func: Callable) -> Callable:
        # Get logger
        if logger_name:
            logger = logging.getLogger(logger_name)
        else:
            logger = logging.getLogger(func.__module__)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Function start logging
            start_time = time.time()
            func_name = func.__name__
            
            # Log function entry
            logger.log(level, f"🚀 Starting function: {func_name}")
            
            # Log arguments if enabled
            if include_args:
                args_str = ", ".join([repr(arg) for arg in args])
                kwargs_str = ", ".join([f"{k}={repr(v)}" for k, v in kwargs.items()])
                all_args = ", ".join(filter(None, [args_str, kwargs_str]))
                if all_args:
                    logger.log(level, f"📥 Arguments: {all_args}")
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                
                # Calculate execution time
                execution_time = time.time() - start_time
                
                # Log successful completion
                logger.log(level, f"✅ Function {func_name} completed successfully in {execution_time:.4f}s")
                
                # Log return value if enabled
                if include_return:
                    logger.log(level, f"📤 Return value: {repr(result)}")
                
                return result
                
            except Exception as e:
                # Calculate execution time
                execution_time = time.time() - start_time
                
                # Log error
                logger.error(f"❌ Function {func_name} failed after {execution_time:.4f}s")
                logger.error(f"💥 Error: {type(e).__name__}: {str(e)}")
                
                # Log traceback if enabled
                if include_traceback:
                    logger.error(f"🔍 Traceback:\n{traceback.format_exc()}")
                
                # Re-raise the exception
                raise
        
        return wrapper
    
    return decorator


def simple_logger(func: Callable) -> Callable:
    """
    A simple decorator that logs basic function execution info.
    
    Args:
        func: Function to decorate
    
    Returns:
        Decorated function with basic logging
    
    Example:
        @simple_logger
        def my_function():
            return "hello"
    """
    return function_logger()(func)


def debug_logger(func: Callable) -> Callable:
    """
    A debug-level decorator that logs detailed function execution info.
    
    Args:
        func: Function to decorate
    
    Returns:
        Decorated function with debug logging
    
    Example:
        @debug_logger
        def my_function():
            return "hello"
    """
    return function_logger(level=logging.DEBUG)(func)


def error_logger(func: Callable) -> Callable:
    """
    A decorator that only logs errors and execution time.
    
    Args:
        func: Function to decorate
    
    Returns:
        Decorated function with error-only logging
    
    Example:
        @error_logger
        def my_function():
            return "hello"
    """
    return function_logger(
        level=logging.ERROR,
        include_args=False,
        include_return=False
    )(func)


# Utility function to set up custom logging
def setup_logging(
    level: int = logging.INFO,
    format_string: Optional[str] = None,
    log_file: Optional[str] = None
) -> None:
    """
    Set up custom logging configuration.
    
    Args:
        level: Logging level
        format_string: Custom format string for log messages
        log_file: Optional file path to write logs to
    
    Example:
        setup_logging(
            level=logging.DEBUG,
            format_string='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            log_file='app.log'
        )
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format=format_string,
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers
    )


# Example usage and testing
if __name__ == "__main__":
    # Example functions to demonstrate the decorators
    
    @function_logger()
    def add_numbers(a: int, b: int) -> int:
        """Add two numbers together."""
        return a + b
    
    @function_logger(level=logging.DEBUG, include_args=False)
    def get_current_time() -> str:
        """Get current time as string."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    @simple_logger
    def divide_numbers(a: float, b: float) -> float:
        """Divide two numbers."""
        return a / b
    
    @error_logger
    def risky_function() -> str:
        """A function that might raise an exception."""
        import random
        if random.random() > 0.5:
            raise ValueError("Random error occurred!")
        return "Success!"
    
    # Test the decorators
    print("Testing function_logger:")
    result = add_numbers(5, 3)
    print(f"Result: {result}\n")
    
    print("Testing debug_logger:")
    time_str = get_current_time()
    print(f"Time: {time_str}\n")
    
    print("Testing simple_logger:")
    result = divide_numbers(10, 2)
    print(f"Result: {result}\n")
    
    print("Testing error_logger:")
    try:
        risky_function()
    except ValueError:
        print("Caught expected error\n")
    
    print("Testing error_logger (success case):")
    try:
        risky_function()
    except ValueError:
        print("Caught expected error")
