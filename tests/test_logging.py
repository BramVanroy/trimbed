"""The package logger: one handler, one namespace, no propagation to the root logger."""

import logging

from trimbed._logging import LOGGER_NAME, configure_logging, get_logger


def test_the_package_logger_is_returned_unqualified():
    assert get_logger().name == LOGGER_NAME
    assert get_logger(LOGGER_NAME).name == LOGGER_NAME


def test_module_names_become_children_of_the_package_logger():
    # Callers pass __name__, which is already prefixed; the prefix must not be doubled.
    assert get_logger("trimbed.selection").name == "trimbed.selection"
    assert get_logger("selection").name == "trimbed.selection"


def test_configuring_twice_does_not_stack_handlers(capsys):
    configure_logging()
    configure_logging(verbose=True)
    logger = logging.getLogger(LOGGER_NAME)

    assert len(logger.handlers) == 1
    assert logger.level == logging.DEBUG
    # Library logs must not reach the root handler and duplicate a script's own output.
    assert logger.propagate is False

    logger.debug("hello")
    assert "hello" in capsys.readouterr().err


def test_quiet_silences_everything_below_a_warning(capsys):
    configure_logging(quiet=True)
    logger = logging.getLogger(LOGGER_NAME)

    logger.info("chatter")
    logger.warning("trouble")

    stderr = capsys.readouterr().err
    assert "chatter" not in stderr
    assert "trouble" in stderr
