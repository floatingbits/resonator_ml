from app.bootstrap import configure_stdout
from app.cli import build_parser, parse_set_arguments
from app.config.app import Config as AppConfig
from app.commands import train, out
from app.config_merge import merge_dataclass


def main():
    parser = build_parser()
    args = parser.parse_args()

    config = AppConfig()

    overrides = parse_set_arguments(args.set)
    merge_dataclass(config, overrides)

    configure_stdout(config,args.command)
    if args.command == "train":
        train.run(config)
    if args.command == "out":
        out.run(config)


if __name__ == "__main__":
    main()
