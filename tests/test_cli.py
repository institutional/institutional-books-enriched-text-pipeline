"""
Tests for the top-level CLI dispatcher (commands/cli.py).
"""

import subprocess
import sys

import click
from click.testing import CliRunner

from commands.cli import COMMANDS, cli


class TestDispatcher:
    def test_help_lists_all_commands(self):
        result = CliRunner().invoke(cli, ["--help"])
        assert result.exit_code == 0
        for name in COMMANDS:
            assert name in result.output

    def test_group_help_stays_lazy(self):
        """Rendering group help must not import heavyweight command modules."""
        # Run in a fresh interpreter: sys.modules in the pytest process may
        # already hold torch/datasets from other test files.
        code = (
            "import sys\n"
            "from click.testing import CliRunner\n"
            "from commands.cli import cli\n"
            "result = CliRunner().invoke(cli, ['--help'])\n"
            "assert result.exit_code == 0\n"
            "assert 'torch' not in sys.modules\n"
            "assert 'datasets' not in sys.modules\n"
        )
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_get_command_resolves_to_click_command(self):
        ctx = click.Context(cli)
        command = cli.get_command(ctx, "dedup-annotate")
        assert isinstance(command, click.Command)

    def test_unknown_command_fails(self):
        result = CliRunner().invoke(cli, ["no-such-command"])
        assert result.exit_code != 0
        assert "No such command" in result.output

    def test_subcommand_help(self):
        result = CliRunner().invoke(cli, ["dedup-annotate", "--help"])
        assert result.exit_code == 0
        assert "--shard-file" in result.output
