"""The command surface: one module per `trimbed` subcommand, plus the router.

The router is [`trimbed.cli.__main__`][trimbed.cli.__main__].
Each module holds a `run(...)` with the logic and an `add_arguments(parser)` describing
its command line, so a command is equally usable from the shell and as a plain Python
call.
"""
