"""
cli.py - top-level dispatcher for pipeline commands

Exposes every command in `commands/` as a subcommand of a single Click group,
so the whole pipeline can be driven as `uv run pipeline.py <command> ...`.

Subcommands are imported lazily: several command modules pull in heavy
dependencies (torch, datasets, classifiers) at import time, and loading all of
them just to render `--help` or run an unrelated command would take seconds.
Help text for the group is therefore kept static in COMMANDS rather than read
from the modules.

Institutional Books - Enriched Text - 2026
"""

import importlib

import click

# name -> (module, short help). Order here is pipeline order and is preserved
# in `--help` output.
COMMANDS: dict[str, tuple[str, str]] = {
    "prepare-shards": (
        "commands.prepare_shards",
        "Download books from HuggingFace and partition into shards",
    ),
    "setup-pipeline": (
        "commands.setup_pipeline",
        "Train Ngram and Nupunkt models from sampled books",
    ),
    "process-shard": (
        "commands.process_shard",
        "Run main processing steps 1-10 on a shard",
    ),
    "compute-bpb": (
        "commands.compute_bpb",
        "Compute per-paragraph bits-per-byte scores (GPU recommended)",
    ),
    "dedup-compute-simhashes": (
        "commands.dedup_compute_simhashes",
        "Compute simhashes for all paragraphs in a shard",
    ),
    "dedup-find-duplicates": (
        "commands.dedup_find_duplicates",
        "Find duplicate clusters across all shards",
    ),
    "dedup-build-lookup": (
        "commands.dedup_build_lookup",
        "Invert clusters.json into per-shard lookup sidecars",
    ),
    "dedup-annotate": (
        "commands.dedup_annotate",
        "Annotate books in a shard with duplicate information",
    ),
    "dedup-mark-removed": (
        "commands.dedup_mark_removed",
        "Mark paragraphs from specified clusters for removal",
    ),
    "postprocess-shard": (
        "commands.postprocess_shard",
        "Run post-processing steps 13-15 on a shard",
    ),
}


class LazyGroup(click.Group):
    """Click group that imports a command's module only when it is invoked."""

    def list_commands(self, ctx: click.Context) -> list[str]:
        return list(COMMANDS)

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        entry = COMMANDS.get(cmd_name)
        if entry is None:
            return None
        module = importlib.import_module(entry[0])
        return module.main

    def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        # Use the static help strings instead of importing every module.
        rows = [(name, short_help) for name, (_, short_help) in COMMANDS.items()]
        with formatter.section("Commands"):
            formatter.write_dl(rows)


@click.command(cls=LazyGroup)
def cli() -> None:
    """
    Institutional Books Enriched Text Pipeline.

    Each subcommand is also runnable directly as a module, e.g.
    `python -m commands.process_shard`.
    """


if __name__ == "__main__":
    cli()
