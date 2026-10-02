"""Run the registered FI-TW experiment profiles, with resumable searches."""

import argparse
import concurrent.futures
import os
import subprocess
import sys
from pathlib import Path

PROFILES = {
    "chronological": [],
    "count-consistent": ["--day-of-month", "--keep-cloud"],
    "historical": ["--day-of-month", "--keep-cloud", "--protocol", "historical-random",
                   "--data", "data/legacy/oulu_features.parquet"],
    "chronological-author": [],
    "historical-author": ["--day-of-month", "--keep-cloud", "--protocol", "historical-random",
                          "--data", "data/legacy/oulu_features.parquet", "--legacy-weather-scaling"],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", nargs="+", choices=list(PROFILES),
                        default=["chronological-author", "historical-author"])
    parser.add_argument("--candidates", type=int, default=100)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--prepare", action="store_true", help="Re-extract raw inputs even when cached")
    args = parser.parse_args()
    if min(args.workers, args.jobs, args.candidates) < 1:
        parser.error("Workers, jobs and candidates must be positive")
    root = Path(__file__).resolve().parent
    os.chdir(root)
    Path("results").mkdir(exist_ok=True)
    if args.prepare or not Path("data/oulu_features.parquet").exists():
        subprocess.run([sys.executable, "-m", "fitw.prepare"], check=True)
    if any(p.startswith("historical") for p in args.profiles) and (args.prepare or not Path("data/legacy/oulu_features.parquet").exists()):
        subprocess.run([sys.executable, "-m", "fitw.legacy"], check=True)
    def run_scenario(profile, scenario):
        log = Path("results") / f"{profile}-{scenario}.log"
        python = str(root / ".venv-author/bin/python") if profile.endswith("-author") else sys.executable
        if not Path(python).exists():
            raise ValueError("Create .venv-author and install requirements-author.txt before running author profiles")
        command = [python, "-m", "fitw.experiment", "--scenario", scenario,
                   "--output", str(Path("results") / profile), "--candidates", str(args.candidates),
                   "--jobs", str(args.jobs)] + PROFILES[profile]
        print(f"Running {profile}/{scenario}; log: {log}", flush=True)
        with log.open("a") as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
        print(f"Completed {profile}/{scenario}", flush=True)
    for profile in args.profiles:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_scenario, profile, scenario) for scenario in ["full", "instant", "categories"]]
            for future in futures:
                future.result()
    subprocess.run([sys.executable, "-m", "fitw.report"], check=True)
    subprocess.run([sys.executable, "-m", "fitw.verify"], check=True)


if __name__ == "__main__":
    main()
