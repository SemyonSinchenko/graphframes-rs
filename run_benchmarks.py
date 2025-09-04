import argparse
import os
import pathlib
import requests
import subprocess
import sys
import shutil
import time

# The base URL for downloading Graphalytics datasets.
BASE_URL = "https://datasets.ldbcouncil.org/graphalytics"

# The local directory where benchmark data will be stored.
BENCH_DATA_DIR = pathlib.Path("benches") / "data" / "ldbc"


def prepare_dataset(dataset_name: str):
    """
    Ensures the dataset is downloaded, decompressed, renamed, and ready for use.
    """
    dataset_dir = BENCH_DATA_DIR / dataset_name
    archive_path = BENCH_DATA_DIR / dataset_name / f"{dataset_name}.tar.zst"
    tar_path = BENCH_DATA_DIR / dataset_name / f"{dataset_name}.tar"

    # If the final extracted directory exists, we are ready to run benchmarks.
    if dataset_dir.is_dir():
        print(f"Dataset '{dataset_name}' is ready.")
        return

    # make dataset_dir if doesn't exist
    os.mkdir(dataset_dir)

    # If the archive doesn't exist, download it.
    if not archive_path.exists():
        print(f"Dataset archive '{archive_path}' not found. Downloading...")
        archive_url = f"{BASE_URL}/{dataset_name}.tar.zst"
        # 3 tries to download the dataset before actually failing
        retries = 3
        for attempt in range(retries):
            try:
                response = requests.get(archive_url, stream=True)
                response.raise_for_status()
                with open(archive_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"Successfully downloaded {archive_path}")
                break  # Success, exit the loop
            except requests.exceptions.RequestException as e:
                print(f"Attempt {attempt + 1} of {retries} failed: {e}", file=sys.stderr)
                if attempt < retries - 1:
                    print("Retrying in 5 seconds...", file=sys.stderr)
                    time.sleep(5)
                else:
                    print(
                        f"Error: Failed to download dataset from {archive_url} after {retries} attempts.",
                        file=sys.stderr,
                    )
                    sys.exit(1)

    # Now, decompress and extract the archive using command-line tools.
    print("Decompressing dataset...")

    # Check for required commands.
    if not shutil.which("unzstd"):
        print(
            "Error: 'unzstd' command not found. Please install zstandard.",
            file=sys.stderr,
        )
        sys.exit(1)
    if not shutil.which("tar"):
        print("Error: 'tar' command not found.", file=sys.stderr)
        sys.exit(1)

    try:
        # Decompress .zst file using unzstd.
        print(f"Running: unzstd -f {archive_path}")
        subprocess.run(
            ["unzstd", "-f", str(archive_path)], check=True, capture_output=True
        )

        # Extract .tar file.
        print(f"Running: tar -xf {tar_path} -C {dataset_dir}")
        subprocess.run(
            ["tar", "-xf", str(tar_path), "-C", str(dataset_dir)],
            check=True,
            capture_output=True,
        )

        # Clean up the intermediate .tar file.
        print(f"Cleaning up {tar_path}")
        os.remove(tar_path)

        print("Decompression and extraction complete.")

        # Rename data files to add .csv extension.
        print(f"Renaming files in {dataset_dir} to add .csv extension...")
        for dirpath, _, filenames in os.walk(dataset_dir):
            for filename in filenames:
                if (not filename.endswith(".properties")) and (
                    not filename.endswith(".tar.zst")
                ):
                    old_path = pathlib.Path(dirpath) / filename
                    new_path = old_path.with_name(f"{old_path.name}.csv")
                    print(f"\tRenaming {old_path} to {new_path}")
                    os.rename(old_path, new_path)
        print("File renaming complete.")

    except subprocess.CalledProcessError as e:
        print(f"Error during decompression: {e}", file=sys.stderr)
        print(f"Stdout: {e.stdout.decode() if e.stdout else 'N/A'}", file=sys.stderr)
        print(f"Stderr: {e.stderr.decode() if e.stderr else 'N/A'}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print(
            f"Error: Could not find intermediate file {tar_path} for cleanup.",
            file=sys.stderr,
        )
        sys.exit(1)


def run_benchmarks(dataset_name: str, checkpoint_interval: int, benchmark_name: str):
    """
    Runs the Rust benchmarks using 'cargo bench', passing the dataset name
    as an environment variable.
    """
    print(f"\nRunning benchmarks for dataset: {dataset_name}")

    # Set the dataset name in an environment variable for the benchmark process.
    env = os.environ.copy()
    env["BENCHMARK_DATASET"] = dataset_name
    env["CHECKPOINT_INTERVAL"] = checkpoint_interval

    # Execute 'cargo bench' and stream its output.
    try:
        cmd = (
            ["cargo", "bench"]
            if not benchmark_name
            else ["cargo", "bench", "--bench", benchmark_name]
        )
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Read and print output line by line.
        for line in iter(process.stdout.readline, ""):
            print(line, end="")

        process.stdout.close()
        return_code = process.wait()

        if return_code != 0:
            print(
                f"\nError: Benchmark process failed with exit code {return_code}",
                file=sys.stderr,
            )
            sys.exit(return_code)

    except FileNotFoundError:
        print(
            "Error: 'cargo' command not found. Is Rust installed and in your PATH?",
            file=sys.stderr,
        )
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error running benchmarks: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """
    Main function to parse arguments and orchestrate the benchmark run.
    """
    parser = argparse.ArgumentParser(
        description="A Python script to download datasets and run GraphFrame benchmarks."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="The name of the Graphalytics dataset to download and use for benchmarking (e.g., 'test-pr-directed').",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=str,
        default="1",
        required=False,
        help="Providing checkpoint_interval to be used in algorithms to run benchmark.",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=False,
        help="Name of the benchmark that needs to run.",
    )
    args = parser.parse_args()
    dataset = args.dataset
    checkpoint_interval = args.checkpoint_interval
    benchmark_name = args.name

    # Ensure the base data directory exists.
    BENCH_DATA_DIR.mkdir(parents=True, exist_ok=True)

    prepare_dataset(dataset)
    run_benchmarks(dataset, checkpoint_interval, benchmark_name)


if __name__ == "__main__":
    main()
