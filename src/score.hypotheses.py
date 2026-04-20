#!/usr/bin/env python3

import enum
import json
import logging
import os
import sys
from argparse import Namespace
from dataclasses import asdict, dataclass
from typing import Sequence

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

import simple_parsing
from simple_parsing import choice, field, flag
from simple_parsing.wrappers import dataclass_wrapper

from mbrs import registry
from mbrs.args import ArgumentParser
from mbrs.metrics import Metric, MetricReferenceless, get_metric


class Format(enum.Enum):
    plain = "plain"
    json = "json"


@dataclass
class CommonArguments:
    """Common arguments."""

    # Hypotheses file.
    hypotheses: str = field(positional=True)
    # Sources file.
    sources: str | None = field(default=None, alias=["-s"])
    # References file.
    references: str | None = field(default=None, alias=["-r"], nargs="+")
    # Output format.
    format: Format = choice(Format, default=Format.json)
    # Type of the metric.
    metric: str = field(
        default="bleu",
        metadata={"choices": registry.get_registry(Metric | MetricReferenceless)},
    )
    # No verbose information and report.
    quiet: bool = flag(default=False)
    # Number of digits for values of float point.
    width: int = field(default=1, alias=["-w"])


def get_argparser(args: Sequence[str] | None = None) -> ArgumentParser:
    meta_parser = ArgumentParser(add_help=False, add_config_path_arg=True)
    meta_parser.add_arguments(CommonArguments, "common")
    for _field in meta_parser._wrappers[0].fields:
        _field.required = False
    known_args, _ = meta_parser.parse_known_args(args=args)

    parser = ArgumentParser(add_help=True, add_config_path_arg=True)
    parser.add_arguments(CommonArguments, "common")
    parser.add_arguments(
        get_metric(known_args.common.metric).Config, "metric", prefix="metric."
    )
    return parser


def format_argparser() -> ArgumentParser:
    parser = get_argparser()
    parser.preprocess_parser()
    return parser


def main(args: Namespace) -> None:
    with open(args.common.hypotheses, mode="r") as f:
        hypotheses = f.readlines()
    num_sents = len(hypotheses)

    sources = None
    if args.common.sources is not None:
        with open(args.common.sources, mode="r") as f:
            sources = f.readlines()
        num_hyp = num_sents // len(sources)
        original_sources = sources.copy()
        sources = []
        for source in original_sources:
            sources.extend([source] * num_hyp)
        assert num_sents == len(sources)

    references_lists: list[list[str]] | None = None
    if args.common.references is not None:
        references_lists = []
        for references_path in args.common.references:
            with open(references_path, mode="r") as f:
                references = f.readlines()
            num_hyp = num_sents // len(references)
            original_references = references.copy()
            references = []
            for reference in original_references:
                references.extend([reference] * num_hyp)
            assert num_sents == len(references)
            references_lists.append(references)

    metric = get_metric(args.common.metric)(args.metric)

    scores = []
    if isinstance(metric, MetricReferenceless):
        assert sources is not None
        scores = metric.scores(hypotheses, sources=sources)
    else:
        assert references_lists is not None
        scores = metric.scores(hypotheses, references_lists, sources)

    score_dict_list = []
    if type(scores) is dict:
        for i in range(num_sents):
            score_dict = {
                "name": args.common.metric,
                "metric_cfg": asdict(args.metric),
            }
            for k in scores.keys():
                score_dict[k] = float(f"{scores[k][i]:.{args.common.width}f}")
            score_dict_list.append(score_dict)
    elif type(scores) is list:
        for i in range(num_sents):
            score_dict = {
                "name": args.common.metric,
                "metric_cfg": asdict(args.metric),
            }
            score_dict["score"] = float(f"{scores[i]:.{args.common.width}f}")
            score_dict_list.append(score_dict)

    if args.common.format == Format.plain:
        for score in score_dict_list:
            score_str = ", ".join(
                f"{k}: {v}" for k, v in score.items() if k not in ["name", "metric_cfg"]
            )
            print(score_str)
    elif args.common.format == Format.json:
        print(json.dumps(score_dict_list, ensure_ascii=False, indent=2))


def cli_main():
    args = get_argparser().parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()