import argparse
import os
from typing import Union

from dotenv import load_dotenv

from src.agent.backend import Provider, GroqModels, TogetherModels, LocalModels
from src.benchmarks.benchmark_main import Benchmark


def main():
    parser = argparse.ArgumentParser(description="Run benchmark simulations.")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--run_all",
        action="store_true",
        help="Run all available configurations.",
    )
    group.add_argument(
        "--config",
        type=str,
        help="Path to a specific configuration file to run.",
    )
    group.add_argument(
        "--run",
        type=str,
        help="Run a specific named configuration.",
    )

    # Other arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./benchmark_results",
        help="Directory to save benchmark results.",
    )
    parser.add_argument(
        "--use_db",
        action="store_true",
        default=False,
        help="Enable database usage in simulations.",
    )
    parser.add_argument(
        "--use_gui",
        action="store_true",
        default=False,
        help="Enable GUI for simulations.",
    )
    parser.add_argument(
        "--num_simulations",
        type=int,
        default=5,
        help="Number of simulations to run per configuration.",
    )
    parser.add_argument(
        "--save_to_csv_false",
        default=False,
        action="store_true",
        help="Save results to CSV files.",
    )
    parser.add_argument(
        "--backend_provider",
        type=provider,
        choices=list(Provider),
        default=Provider.GROQ,
        help="Specify the backend provider.",
    )
    parser.add_argument(
        "--backend_model",
        type=model,
        choices=list(TogetherModels) + list(GroqModels) + list(LocalModels),
        default=GroqModels.LLAMA_8B,
        help="Specify the backend model to use.",
    )

    args = parser.parse_args()

    # Initialize the Benchmark object
    benchmark = Benchmark(
        use_db=args.use_db,
        use_gui=args.use_gui,
        output_dir=args.output_dir,
        num_simulations=args.num_simulations,
        backend_provider=args.backend_provider,
        backend_model=args.backend_model,
    )

    # Handle the mutually exclusive options
    if args.run_all:
        print("Running all configurations...")
        benchmark.run_all(save_to_csv=args.save_to_csv_false)
    elif args.config:
        if not os.path.exists(args.config):
            print(f"Error: Configuration file '{args.config}' not found.")
            return
        print(f"Running benchmark for configuration file: {args.config}")
        benchmark.from_config(args.config, save_to_csv=args.save_to_csv_false)
    elif args.run:
        print("Running a specific named configuration...")
        benchmark.run(config_keys=[args.run], save_to_csv=not args.save_to_csv_false)
    else:
        print("Error: Unexpected state. Please check the arguments.")
        parser.print_help()


def provider(prov: str) -> Provider:
    """Convert a string to a Provider enum."""
    for p in Provider:
        if p.name.lower() == prov.lower():
            return p
    raise argparse.ArgumentTypeError(f"Invalid provider: {prov}")


def model(model_inp: str) -> Union[GroqModels, TogetherModels, LocalModels]:
    """Convert a string to the corresponding model enum."""
    for provider in [GroqModels, TogetherModels, LocalModels]:
        for m in provider:
            if str(m).lower() == model_inp.lower():
                return m
    raise argparse.ArgumentTypeError(f"Invalid model: {model_inp}")


if __name__ == "__main__":
    load_dotenv(".env")
    main()
