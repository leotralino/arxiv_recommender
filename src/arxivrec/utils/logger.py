import logging
import sys

from rich.box import DOUBLE_EDGE  # Or HEAVY, SQUARE, etc.
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


class CustomFormatter(logging.Formatter):
    MAGENTA = "\033[35m"
    GREEN = "\033[32m"
    BLUE = "\033[34m"
    YELLOW = "\033[33m"
    WHITE = "\033[37m"
    RED = "\033[31m"
    RESET = "\033[0m"

    def format(self, record):
        if record.levelno >= logging.ERROR:
            level_color = self.RED
        elif record.levelno >= logging.WARNING:
            level_color = self.YELLOW
        else:
            level_color = self.WHITE

        fmt = (
            f"{self.MAGENTA}%(asctime)s{self.RESET} | "
            f"{level_color}%(levelname)s{self.RESET} | "
            f"{self.GREEN}%(filename)s{self.RESET}:"
            f"{self.BLUE}%(funcName)s{self.RESET}:"
            f"{self.YELLOW}L%(lineno)d{self.RESET} | - "
            f"{self.WHITE}%(message)s{self.RESET}"
        )

        self._style._fmt = fmt
        return super().format(record)


def setup_logging():
    """Configures the root logger with our custom colorful style."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Prevent duplicate handlers if setup_logging is called twice
    if not root_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = CustomFormatter(datefmt="%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)


def show_topic_table(topic_list):
    console = Console()

    table = Table(
        title="[bold blue]ArXiv Topics to Process[/bold blue]",
        show_header=True,
        header_style="bold white",
        show_lines=True,  # This adds the horizontal lines between rows
        box=DOUBLE_EDGE,  # This adds the outer and inner vertical boundaries
    )

    table.add_column("ID", style="cyan", justify="center")
    table.add_column("Categories", style="green")
    table.add_column(
        "Description", style="magenta", width=50
    )  # Prevents long text from breaking the grid

    for t in topic_list:
        table.add_row(t.id, ", ".join(t.categories), t.description)

    console.print(table)


def show_registry_table(LLM_REGISTRY, NOTIFIER_REGISTRY):
    console = Console()

    llm_text = Text.assemble(
        ("LLMs:      ", "bold white"),
        (", ".join(LLM_REGISTRY.show_available()), "cyan"),
    )
    note_text = Text.assemble(
        ("Notifiers: ", "bold white"),
        (", ".join(NOTIFIER_REGISTRY.show_available()), "green"),
    )

    content = Text.assemble(llm_text, "\n", note_text)

    panel = Panel(
        content,
        title="[bold blue]Registered Modules[/bold blue]",
        subtitle="[dim]Change in config.yaml if needed[/dim]",
        expand=True,
        border_style="bright_blue",
        padding=(1, 2),
    )

    console.print(panel)
