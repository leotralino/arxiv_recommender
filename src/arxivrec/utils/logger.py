import logging
import sys
import types

from loguru import logger
from rich.box import DOUBLE_EDGE
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


class _InterceptHandler(logging.Handler):
    """Route stdlib logging (third-party libs) into loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame: types.FrameType | None = sys._getframe(6)
        depth = 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging(level: str = "INFO") -> None:
    logger.remove()
    logger.add(
        sys.stdout,
        level=level,
        format=(
            "<magenta>{time:YYYY-MM-DD HH:mm:ss}</magenta> | "
            "<level>{level:<8}</level> | "
            "<green>{file}</green>:<cyan>{function}</cyan>:<yellow>L{line}</yellow> | "
            "{message}"
        ),
        colorize=True,
    )
    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)


def show_topic_table(topic_list) -> None:
    console = Console()

    table = Table(
        title="[bold blue]ArXiv Topics to Process[/bold blue]",
        show_header=True,
        header_style="bold white",
        show_lines=True,
        box=DOUBLE_EDGE,
    )

    table.add_column("ID", style="cyan", justify="center")
    table.add_column("Categories", style="green")
    table.add_column("Description", style="magenta", width=50)

    for t in topic_list:
        table.add_row(t.id, ", ".join(t.categories), t.description)

    console.print(table)


def show_registry_table(LLM_REGISTRY, NOTIFIER_REGISTRY) -> None:
    console = Console()

    llm_text = Text.assemble(
        ("LLMs:      ", "bold white"),
        (", ".join(LLM_REGISTRY.show_available()), "cyan"),
    )
    note_text = Text.assemble(
        ("Notifiers: ", "bold white"),
        (", ".join(NOTIFIER_REGISTRY.show_available()), "green"),
    )

    panel = Panel(
        Text.assemble(llm_text, "\n", note_text),
        title="[bold blue]Registered Modules[/bold blue]",
        subtitle="[dim]Change in config.yaml if needed[/dim]",
        expand=True,
        border_style="bright_blue",
        padding=(1, 2),
    )

    console.print(panel)
