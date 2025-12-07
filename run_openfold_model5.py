import os
import time
import subprocess
import argparse
from pathlib import Path

def build_cmd(aln_version, out_version, model_v, rewrite=False):
    cmd = [
        "python", "-u", "run_pretrained_openfold.py",
        "/data4T/ganjh/tcr/tcr_21_fas",
        "data/pdb_mmcif/mmcif_files/",
        "--config_preset", model_v,
        "--model_device", "cuda:0",
        "--output_dir", f"/data4T/ganjh/tcr/openfold_out_{out_version}",
        "--use_precomputed_alignments", f"/data4T/ganjh/tcr/openfold_out/{aln_version}",
        "--data_random_seed", "42"
    ]
    if rewrite:
        cmd.append("--rewrite")
    return cmd


def run_model(model_v, aln_version, out_version, rewrite, dry_run=False):
    cmd = build_cmd(aln_version, out_version, model_v, rewrite)
    log_path = Path(f"openfold_out_{out_version}_{model_v}.log")

    print(f"\n[INFO] Running model: {model_v}")
    print("[CMD]", " ".join(cmd))
    if dry_run:
        return

    tcr_codes = [x[:8] for x in os.listdir("/data4T/ganjh/tcr/tcr_21_fas")]
    files = [
        f"/data4T/ganjh/tcr/openfold_out_{out_version}/predictions/{x[:4]}_{x[5]}-{x[:4]}_{x[-1]}_{model_v}_relaxed.pdb" 
        for x in tcr_codes
    ]
    print(files[0])
    for _ in  range(5):
        missing = [f for f in files if not os.path.exists(f)]
        if not missing:
            print("All prediction files exist. Proceeding...")
            break

        print(f"{len(missing)} files not found. Retrying inference...")
        with log_path.open("w") as log_file:
            subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT)

        time.sleep(5)
        


def main():
    parser = argparse.ArgumentParser(description="Run OpenFold models with config.")
    parser.add_argument("aln_version", help="Alignment version")
    parser.add_argument("out_version", help="Output version")
    parser.add_argument("num", type=int, choices=[1, 5], help="Number of models (1 or 5)")
    parser.add_argument("--rewrite", action="store_true", help="Whether to rewrite existing outputs")
    parser.add_argument("--dry-run", action="store_true", help="Only print commands without running them")
    args = parser.parse_args()

    if args.num == 1:
        run_model("model_1_multimer_v3", args.aln_version, args.out_version, args.rewrite, args.dry_run)
    else:
        for i in range(5):
            model_v = i + 1
            model_name = f"model_{model_v}_multimer_v3"
            run_model(model_name, args.aln_version, args.out_version, args.rewrite, args.dry_run)


if __name__ == "__main__":
    main()
