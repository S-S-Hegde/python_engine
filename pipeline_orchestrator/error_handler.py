import logging

logger = logging.getLogger(__name__)

class PipelineErrorHandler:
    def handle_module_failure(self, module_name: str, exception: Exception):
        """
        Logs and processes failures, deciding if the pipeline should halt or degrade gracefully.
        """
        logger.error(f"Module {module_name} failed: {str(exception)}")
        # In this implementation, we allow graceful degradation by bubbling up to the runner.
        # Future expansions can map specific exception types to retry logic.
        return str(exception)
