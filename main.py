import argparse

from pathlib import Path

from cross_validation import MyCrossValidation
from utils import install_logger, load_config, logger
from utils.tool import seed_everything


def init_logger(args):
    install_logger(time_format="[%X]", hook_exceptions=False)

    log_dir: Path = args.logging.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loggings will be saved at {}", log_dir)

    logger.add(
        log_dir / 'run_{time}.log',
        format=
            "{time:YYYY-MM-DD HH:mm:ss.SSS} "
            "{level: <8} {module}:{function}:{line} - {message}",
        rotation="100 MB",
        retention="1 years"
    )

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default='./config/index.yaml', help="Path to the config file.")
    parser.add_argument("--folds", type=int, nargs="+", help="Held-out subjects to run (all subjects by default).")
    parser.add_argument("--prepare-only", action="store_true", help="Build/validate feature cache without training.")
    parser.add_argument("--rebuild-data", action="store_true", help="Rebuild features after changing raw recordings.")
    parser.add_argument("--evaluate", action="store_true", help="Re-evaluate saved fold checkpoints without training.")
    cli = parser.parse_args()
    return load_config(cli.config), cli

def show_args(args):
    logger.info("Parsed Arguments:\n{}", args)

def main():
    args, cli = parse_args()

    # init logger
    init_logger(args)

    # setup random seed
    seed_everything(args.reproduce.random_seed, args.reproduce.deterministic)

    show_args(args)

    cv = MyCrossValidation(args)
    if cli.rebuild_data:
        cv.datapipe.build()
    else:
        cv.ensure_data_prepared()
    if cli.evaluate:
        cv.evaluate(output_dir=args.training.records_dir / "evaluation", subjects=cli.folds)
    elif not cli.prepare_only:
        cv.leave_one_sub_out(subjects=cli.folds)


if __name__ == '__main__':
    main()
