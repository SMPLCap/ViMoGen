"""multi process logger, adapted from accelerate logging."""
import functools
import logging
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from parallel.utils import on_main_process


class MultiProcessAdapter(logging.LoggerAdapter):

    @staticmethod
    def _should_log(main_process_only):
        """Check if log should be performed."""
        if not dist.is_available() or not dist.is_initialized():
            is_main_process = True
        else:
            is_main_process = dist.get_rank() == 0
        return not main_process_only or (main_process_only and is_main_process)

    def log(self, level, msg, *args, **kwargs):
        """Delegates logger call after checking if we should log.

        Accepts a new kwarg of `main_process_only`, which will dictate whether it will be logged across all processes
        or only the main executed one. Default is `True` if not passed

        Also accepts "in_order", which if `True` makes the processes log one by one, in order. This is much easier to
        read, but comes at the cost of sometimes needing to wait for the other processes. Default is `False` to not
        break with the previous behavior.

        `in_order` is ignored if `main_process_only` is passed.
        """
        main_process_only = kwargs.pop('main_process_only', True)
        in_order = kwargs.pop('in_order', False)
        # set `stacklevel` to exclude ourself in `Logger.findCaller()` while respecting user's choice
        kwargs.setdefault('stacklevel', 2)

        if self.isEnabledFor(level):
            if self._should_log(main_process_only):
                msg, kwargs = self.process(msg, kwargs)
                self.logger.log(level, msg, *args, **kwargs)

            elif in_order:
                if not dist.is_available() or not dist.is_initialized():
                    msg, kwargs = self.process(msg, kwargs)
                    self.logger.log(level, msg, *args, **kwargs)
                else:
                    rank = dist.get_rank()
                    num_processes = dist.get_world_size()
                    for i in range(num_processes):
                        if i == rank:
                            msg, kwargs = self.process(msg, kwargs)
                            self.logger.log(level, msg, *args, **kwargs)
                        dist.barrier()

    @functools.lru_cache(None)
    def warning_once(self, *args, **kwargs):
        """This method is identical to `logger.warning()`, but will emit the
        warning with the same message only once.

        Note: The cache is for the function arguments, so 2 different callers using the same arguments will hit the
        cache. The assumption here is that all warning messages are unique across the code. If they aren't then need to
        switch to another type of cache that includes the caller frame information in the hashing function.
        """
        self.warning(*args, **kwargs)


def get_logger(name: str, log_level: str = 'DEBUG'):
    """Returns a `logging.Logger` for `name` that can handle
    multiprocessing."""
    logger = logging.getLogger(name)
    if log_level is not None:
        logger.setLevel(log_level.upper())
        logger.root.setLevel(log_level.upper())
    return MultiProcessAdapter(logger, {})


class TbTracker:

    name = 'tensorboard'
    requires_logging_directory = True
    main_process_only = True

    @on_main_process
    def __init__(self, logging_dir, **kwargs):
        super().__init__()
        self.logging_dir = logging_dir
        self.writer = SummaryWriter(self.logging_dir, **kwargs)

    @property
    def tracker(self):
        return self.writer

    @on_main_process
    def add_scalar(self, tag, scalar_value, **kwargs):
        self.writer.add_scalar(tag=tag, scalar_value=scalar_value, **kwargs)

    @on_main_process
    def add_text(self, tag, text_string, **kwargs):
        self.writer.add_text(tag=tag, text_string=text_string, **kwargs)

    @on_main_process
    def add_figure(self, tag, figure, **kwargs):
        self.writer.add_figure(tag=tag, figure=figure, **kwargs)
