from app.bootstrap import configure_stdout
from app.cli import build_parser, parse_set_arguments
from app.config.app import Config as AppConfig
from app.commands import train, out, plot_weights, plot_training_data, plot_training_result, plot_spectrum_comparison
from app.config_merge import merge_dataclass


def main():
    command_dict = {
        "train": train,
        "out": out,
        "plot_weights": plot_weights,
        "plot_training_data": plot_training_data,
        "plot_training_result": plot_training_result,
        "plot_spectrum_comparison": plot_spectrum_comparison
    }
    parser = build_parser(list(command_dict.keys()))
    args = parser.parse_args()

    config = AppConfig()

    overrides = parse_set_arguments(args.set)
    merge_dataclass(config, overrides)

    configure_stdout(config,args.command)

    if command_dict[args.command]:
        command_dict[args.command].run(config)


if __name__ == "__main__":
    main()
