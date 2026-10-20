import argparse
import json
import sys
from pathlib import Path

from src.glucotwin.data.pipeline import (
    compute_dataset_summary,
    extract_features,
    generate_splits,
    harmonize_data,
    ingest_raw_data,
    load_config,
    run_data_pipeline,
)
from src.glucotwin.models.cv import CrossValidationManager
from src.glucotwin.models.evaluator import CrossValidationEvaluator
from src.glucotwin.models.trainer import HyperparameterTuner

try:
    import typer
    from rich.console import Console

    HAS_TYPER = True
except ImportError:
    HAS_TYPER = False

if HAS_TYPER:
    app = typer.Typer(help="GlucoTwin CLI: India-aware Digital Twin for Type 2 Diabetes Management")
    data_app = typer.Typer(
        help="Data ingestion, harmonization, feature engineering, and splitting commands"
    )
    models_app = typer.Typer(
        help="Model training, tuning, cross-validation, and evaluation commands"
    )

    app.add_typer(data_app, name="data")
    app.add_typer(models_app, name="models")
    console = Console()

    @app.command()
    def version():
        typer.echo("GlucoTwin v0.1.0")

    @data_app.command("ingest")
    def data_ingest(
        raw_dir: str = typer.Option("data/raw", help="Path to raw data directory"),
        interim_dir: str = typer.Option("data/interim", help="Path to interim data directory"),
        include_hupa: bool = typer.Option(False, help="Include HUPA-UCM T1D cohort"),
        config: str = typer.Option("params.yaml", help="Path to configuration file"),
    ):
        cfg = load_config(config)
        hba1c_cfg = cfg.get("hba1c_cutoffs", {})
        stats = ingest_raw_data(
            raw_dir=Path(raw_dir),
            interim_dir=Path(interim_dir),
            include_hupa_t1d=include_hupa,
            hba1c_cutoffs=hba1c_cfg,
        )
        console.print("[green]Ingestion complete:[/green]", stats)

    @data_app.command("harmonize")
    def data_harmonize(
        interim_dir: str = typer.Option("data/interim", help="Path to interim data directory"),
        processed_dir: str = typer.Option(
            "data/processed", help="Path to processed data directory"
        ),
        config: str = typer.Option("params.yaml", help="Path to configuration file"),
    ):
        cfg = load_config(config)
        harm_cfg = cfg.get("harmonization", {})
        stats = harmonize_data(
            interim_dir=Path(interim_dir),
            processed_dir=Path(processed_dir),
            harmonization_cfg=harm_cfg,
        )
        console.print("[green]Harmonization complete:[/green]", stats)

    @data_app.command("features")
    def data_features(
        processed_dir: str = typer.Option(
            "data/processed", help="Path to processed data directory"
        ),
        config: str = typer.Option("params.yaml", help="Path to configuration file"),
    ):
        cfg = load_config(config)
        feat_cfg = cfg.get("features", {})
        dataset = extract_features(
            processed_dir=Path(processed_dir),
            feature_cfg=feat_cfg,
        )
        console.print(f"[green]Extracted {len(dataset)} feature windows.[/green]")

    @data_app.command("summary")
    def data_summary(
        processed_dir: str = typer.Option(
            "data/processed", help="Path to processed data directory"
        ),
    ):
        summary = compute_dataset_summary(Path(processed_dir))
        console.print("[bold green]Dataset Summary across sources:[/bold green]")
        console.print(json.dumps(summary, indent=2))

    @data_app.command("split")
    def data_split(
        processed_dir: str = typer.Option(
            "data/processed", help="Path to processed data directory"
        ),
        config: str = typer.Option("params.yaml", help="Path to configuration file"),
        seed: int = typer.Option(42, help="Random seed for patient splitting"),
    ):
        cfg = load_config(config)
        split_cfg = cfg.get("splits", {})
        splits, manifest = generate_splits(
            processed_dir=Path(processed_dir),
            split_cfg=split_cfg,
            random_seed=seed,
        )
        console.print("[green]Patient splits generated:[/green]")
        console.print(json.dumps(manifest, indent=2))

    @data_app.command("run-all")
    def data_run_all(
        config: str = typer.Option("params.yaml", help="Path to configuration file"),
    ):
        results = run_data_pipeline(config_path=config)
        console.print("[bold green]Data pipeline completed successfully:[/bold green]")
        console.print(json.dumps(results, indent=2, default=str))

    @models_app.command("tune")
    def models_tune(
        processed_dir: str = typer.Option("data/processed", help="Path to processed data"),
        n_trials: int = typer.Option(15, help="Number of Optuna trials"),
        output_file: str = typer.Option(
            "models/checkpoints/best_hyperparams.json", help="Output path"
        ),
    ):
        cv = CrossValidationManager(processed_dir=Path(processed_dir))
        fold0 = cv.get_fold_data(0)
        tuner = HyperparameterTuner(fold0, n_trials=n_trials)
        console.print(f"[bold cyan]Tuning LightGBM with {n_trials} trials...[/bold cyan]")
        best_lgb = tuner.tune_lightgbm()
        console.print("[green]Best LightGBM parameters:[/green]", best_lgb)

        console.print(f"[bold cyan]Tuning GRU with {n_trials} trials...[/bold cyan]")
        best_gru = tuner.tune_gru()
        console.print("[green]Best GRU parameters:[/green]", best_gru)

        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fp:
            json.dump({"lightgbm": best_lgb, "gru": best_gru}, fp, indent=2)
        console.print(f"[bold green]Saved best parameters to {output_file}[/bold green]")

    @models_app.command("cv")
    def models_cv(
        processed_dir: str = typer.Option("data/processed", help="Path to processed data"),
        quick: bool = typer.Option(False, "--quick", help="Run quick 1-fold development mode"),
    ):
        cv = CrossValidationManager(processed_dir=Path(processed_dir))
        evaluator = CrossValidationEvaluator(cv, quick_mode=quick)
        console.print(
            f"[bold cyan]Executing {'Quick' if quick else '5-Fold'} Leave-Patients-Out Cross-Validation...[/bold cyan]"
        )
        summary = evaluator.run_evaluation()
        console.print("[bold green]Cross-Validation Complete! Summary metrics:[/bold green]")
        console.print(json.dumps(summary, indent=2))

    @models_app.command("run-all")
    def models_run_all(
        processed_dir: str = typer.Option("data/processed", help="Path to processed data"),
        quick: bool = typer.Option(False, "--quick", help="Run quick mode"),
    ):
        cv = CrossValidationManager(processed_dir=Path(processed_dir))
        cv.save_manifest()
        evaluator = CrossValidationEvaluator(cv, quick_mode=quick)
        summary = evaluator.run_evaluation()
        console.print(
            "[bold green]All models trained and evaluated. Artifacts generated in reports/ and models/checkpoints/[/bold green]"
        )
        console.print(json.dumps(summary, indent=2))


def main():
    if HAS_TYPER and len(sys.argv) > 1 and not sys.argv[1].startswith("--help-argparse"):
        app()
    else:
        parser = argparse.ArgumentParser(description="GlucoTwin CLI")
        subparsers = parser.add_subparsers(dest="command")

        data_parser = subparsers.add_parser("data", help="Data operations")
        data_subparsers = data_parser.add_subparsers(dest="subcommand")

        ingest_parser = data_subparsers.add_parser("ingest", help="Ingest raw data")
        ingest_parser.add_argument("--raw-dir", default="data/raw")
        ingest_parser.add_argument("--interim-dir", default="data/interim")
        ingest_parser.add_argument("--include-hupa", action="store_true", default=False)
        ingest_parser.add_argument("--config", default="params.yaml")

        harm_parser = data_subparsers.add_parser("harmonize", help="Harmonize data")
        harm_parser.add_argument("--interim-dir", default="data/interim")
        harm_parser.add_argument("--processed-dir", default="data/processed")
        harm_parser.add_argument("--config", default="params.yaml")

        feat_parser = data_subparsers.add_parser("features", help="Extract features")
        feat_parser.add_argument("--processed-dir", default="data/processed")
        feat_parser.add_argument("--config", default="params.yaml")

        summary_parser = data_subparsers.add_parser("summary", help="Compute dataset summary")
        summary_parser.add_argument("--processed-dir", default="data/processed")

        split_parser = data_subparsers.add_parser("split", help="Generate patient splits")
        split_parser.add_argument("--processed-dir", default="data/processed")
        split_parser.add_argument("--config", default="params.yaml")
        split_parser.add_argument("--seed", type=int, default=42)

        run_all_parser = data_subparsers.add_parser("run-all", help="Run full pipeline")
        run_all_parser.add_argument("--config", default="params.yaml")

        models_parser = subparsers.add_parser("models", help="Model operations")
        models_subparsers = models_parser.add_subparsers(dest="subcommand")

        cv_parser = models_subparsers.add_parser("cv", help="Run cross-validation")
        cv_parser.add_argument("--processed-dir", default="data/processed")
        cv_parser.add_argument("--quick", action="store_true", default=False)

        tune_parser = models_subparsers.add_parser("tune", help="Tune hyperparameters")
        tune_parser.add_argument("--processed-dir", default="data/processed")
        tune_parser.add_argument("--n-trials", type=int, default=15)

        args = parser.parse_args()

        if args.command == "data":
            if args.subcommand == "ingest":
                cfg = load_config(args.config)
                hba1c_cfg = cfg.get("hba1c_cutoffs", {})
                stats = ingest_raw_data(
                    Path(args.raw_dir), Path(args.interim_dir), args.include_hupa, hba1c_cfg
                )
                print(f"Ingestion complete: {stats}")
            elif args.subcommand == "harmonize":
                cfg = load_config(args.config)
                stats = harmonize_data(
                    Path(args.interim_dir), Path(args.processed_dir), cfg.get("harmonization", {})
                )
                print(f"Harmonization complete: {stats}")
            elif args.subcommand == "features":
                cfg = load_config(args.config)
                ds = extract_features(Path(args.processed_dir), cfg.get("features", {}))
                print(f"Extracted {len(ds)} feature windows.")
            elif args.subcommand == "summary":
                summary = compute_dataset_summary(Path(args.processed_dir))
                print(json.dumps(summary, indent=2))
            elif args.subcommand == "split":
                cfg = load_config(args.config)
                splits, manifest = generate_splits(
                    Path(args.processed_dir), cfg.get("splits", {}), args.seed
                )
                print(json.dumps(manifest, indent=2))
            elif args.subcommand == "run-all":
                res = run_data_pipeline(args.config)
                print(json.dumps(res, indent=2, default=str))
            else:
                parser.print_help()
        elif args.command == "models":
            if args.subcommand == "cv":
                cv = CrossValidationManager(processed_dir=Path(args.processed_dir))
                evaluator = CrossValidationEvaluator(cv, quick_mode=args.quick)
                res = evaluator.run_evaluation()
                print(json.dumps(res, indent=2))
            elif args.subcommand == "tune":
                cv = CrossValidationManager(processed_dir=Path(args.processed_dir))
                fold0 = cv.get_fold_data(0)
                tuner = HyperparameterTuner(fold0, n_trials=args.n_trials)
                b_lgb = tuner.tune_lightgbm()
                b_gru = tuner.tune_gru()
                print(json.dumps({"lightgbm": b_lgb, "gru": b_gru}, indent=2))
            else:
                parser.print_help()
        else:
            parser.print_help()


if __name__ == "__main__":
    main()
