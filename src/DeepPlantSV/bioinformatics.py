from __future__ import annotations

import os
import subprocess
from pathlib import Path
from timeit import default_timer as timer
from typing import List, Optional

import pandas as pd
from tqdm.auto import tqdm


def check_tool_exists(tool_path: str, tool_name: str) -> bool:
    try:
        version_arg = "--version" if tool_name != "bwa" else ""
        cmd = [tool_path]
        if version_arg:
            cmd.append(version_arg)
        subprocess.run(cmd, check=(tool_name != "bwa"), capture_output=True, text=True, encoding="utf-8", errors="ignore")
        print(f"check tool ok: {tool_name} -> {tool_path}")
        return True
    except FileNotFoundError:
        print(f"tool not found: {tool_name} ({tool_path})")
        return False
    except subprocess.CalledProcessError as e:
        print(f"failed to check {tool_name}: {e}")
        return False
    except Exception as e:
        print(f"unexpected error while checking {tool_name}: {e}")
        return False


def run_command(command: List[str], step_name: str) -> bool:
    command_str = " ".join(command)
    print(f"--- {step_name} ---")
    print(command_str)
    try:
        process = subprocess.run(command, check=True, capture_output=True, text=True, encoding="utf-8", errors="ignore")
        if process.stderr:
            print(process.stderr[:500])
        return True
    except Exception as e:
        print(f"{step_name} failed: {e}")
        return False


def preprocess_for_prediction(
    input_fasta: str,
    reference_fasta: str,
    padding: int,
    output_dir: str,
    bwa_path: str,
    samtools_path: str,
    bcftools_path: str,
    index_dir: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    print("start preprocessing for prediction")
    start_time = timer()

    if not os.path.exists(input_fasta):
        print(f"missing input FASTA: {input_fasta}")
        return None
    if not os.path.exists(reference_fasta):
        print(f"missing reference FASTA: {reference_fasta}")
        return None
    if index_dir and not os.path.isdir(index_dir):
        print(f"warning: invalid index_dir {index_dir}; will ignore")
        index_dir = None

    os.makedirs(output_dir, exist_ok=True)
    ref_basename = os.path.basename(reference_fasta)
    ref_dir = os.path.dirname(reference_fasta)
    safe_ref_basename = "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in ref_basename)

    output_sam = os.path.join(output_dir, f"{safe_ref_basename}.aligned.sam")
    output_bam = os.path.join(output_dir, f"{safe_ref_basename}.aligned.bam")
    sorted_bam = os.path.join(output_dir, f"{safe_ref_basename}.aligned.sorted.bam")
    vcf_gz_path = os.path.join(output_dir, f"{safe_ref_basename}.variants.vcf.gz")
    output_extracted_fasta = os.path.join(output_dir, f"{safe_ref_basename}.extracted_fragments.fasta")

    bwa_exts = ["amb", "ann", "bwt", "pac", "sa"]
    bwa_indices_found = False
    if index_dir:
        expected_indices = [os.path.join(index_dir, f"{ref_basename}.{ext}") for ext in bwa_exts]
        bwa_indices_found = all(os.path.exists(f) for f in expected_indices)
    if not bwa_indices_found:
        expected_indices = [f"{reference_fasta}.{ext}" for ext in bwa_exts]
        bwa_indices_found = all(os.path.exists(f) for f in expected_indices)
    if not bwa_indices_found:
        if not run_command([bwa_path, "index", reference_fasta], "bwa index"):
            return None

    try:
        with open(output_sam, "w", encoding="utf-8") as f_sam:
            subprocess.run([bwa_path, "mem", reference_fasta, input_fasta], check=True, stdout=f_sam, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"bwa mem failed: {e}")
        return None

    try:
        ps_view = subprocess.Popen([samtools_path, "view", "-bS", output_sam], stdout=subprocess.PIPE)
        ps_sort = subprocess.run([samtools_path, "sort", "-o", sorted_bam], stdin=ps_view.stdout, check=True, capture_output=True, text=True, encoding="utf-8", errors="ignore")
        ps_view.stdout.close()
        retcode = ps_view.wait()
        if retcode:
            raise subprocess.CalledProcessError(retcode, [samtools_path, "view", "-bS", output_sam])
    except Exception as e:
        print(f"samtools view/sort failed: {e}")
        return None

    if not os.path.exists(sorted_bam + ".bai"):
        if not run_command([samtools_path, "index", sorted_bam], "samtools index bam"):
            return None

    try:
        ps_mpileup = subprocess.Popen([bcftools_path, "mpileup", "-Ou", "-f", reference_fasta, sorted_bam], stdout=subprocess.PIPE)
        ps_call = subprocess.run([bcftools_path, "call", "-mv", "-Oz", "-o", vcf_gz_path], stdin=ps_mpileup.stdout, check=True, capture_output=True, text=True, encoding="utf-8", errors="ignore")
        ps_mpileup.stdout.close()
        retcode = ps_mpileup.wait()
        if retcode:
            raise subprocess.CalledProcessError(retcode, [bcftools_path, "mpileup", "-Ou", "-f", reference_fasta, sorted_bam])
    except Exception as e:
        print(f"bcftools mpileup/call failed: {e}")
        return None

    if not (os.path.exists(vcf_gz_path + ".tbi") or os.path.exists(vcf_gz_path + ".csi")):
        if not run_command([bcftools_path, "index", vcf_gz_path], "bcftools index vcf"):
            return None

    fai_found = False
    if index_dir and os.path.exists(os.path.join(index_dir, ref_basename + ".fai")):
        fai_found = True
    if not fai_found and os.path.exists(reference_fasta + ".fai"):
        fai_found = True
    if not fai_found:
        if not run_command([samtools_path, "faidx", reference_fasta], "samtools faidx"):
            return None

    extracted_data = []
    try:
        query_process = subprocess.run([bcftools_path, "query", "-f", "%CHROM\t%POS\t%ID\t%REF\t%ALT\n", vcf_gz_path], check=True, capture_output=True, text=True, encoding="utf-8", errors="ignore")
        variant_lines = query_process.stdout.strip().splitlines()
        if not variant_lines:
            return pd.DataFrame(columns=["id", "sequence", "contig", "position", "ref", "alt"])
        for line in tqdm(variant_lines, desc="extract_variants"):
            if line.startswith("#"):
                continue
            try:
                chrom, pos_str, var_id, ref, alt = line.strip().split("\t")
                pos = int(pos_str)
                start = max(1, pos - padding)
                end = pos + padding
                region = f"{chrom}:{start}-{end}"
                faidx_process = subprocess.run([samtools_path, "faidx", reference_fasta, region], check=False, capture_output=True, text=True, encoding="utf-8", errors="ignore")
                if faidx_process.returncode != 0 or not faidx_process.stdout.strip():
                    continue
                sequence_lines = faidx_process.stdout.strip().splitlines()
                if len(sequence_lines) <= 1:
                    continue
                sequence = "".join(sequence_lines[1:]).upper()
                expected_len = end - start + 1
                if abs(len(sequence) - expected_len) > padding // 2:
                    continue
                unique_id = f"{chrom}_{pos}_{ref}_{alt}" if var_id == "." else f"{var_id}_{chrom}_{pos}"
                extracted_data.append({
                    "id": unique_id,
                    "sequence": sequence,
                    "contig": chrom,
                    "position": pos,
                    "ref": ref,
                    "alt": alt,
                })
            except Exception:
                continue
    except Exception as e:
        print(f"failed to query or parse VCF: {e}")
        return None

    try:
        if os.path.exists(output_sam):
            os.remove(output_sam)
        if os.path.exists(output_bam):
            os.remove(output_bam)
    except OSError:
        pass

    print(f"preprocessing finished in {timer() - start_time:.2f}s")
    if not extracted_data:
        return pd.DataFrame(columns=["id", "sequence", "contig", "position", "ref", "alt"])
    result_df = pd.DataFrame(extracted_data)
    try:
        with open(output_extracted_fasta, "w", encoding="utf-8") as f:
            for _, row in result_df.iterrows():
                f.write(f">{row['id']}\n{row['sequence']}\n")
    except IOError:
        pass
    return result_df[["id", "sequence", "contig", "position", "ref", "alt"]]
