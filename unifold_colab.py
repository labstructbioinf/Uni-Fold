import os
import re
import hashlib
import random
import numpy as np
import argparse
import tarfile
import requests
import time
import logging
import pickle
import gzip
import torch
import json

from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union, Any, Optional
from tqdm import tqdm
from Bio import SeqIO

from unifold.data import residue_constants, protein
from unifold.msa.utils import divide_multi_chains
from unifold.msa import templates, pipeline
from unifold.msa.tools import hhsearch

from unifold.msa import parsers
from unifold.msa import pipeline
from unifold.data.utils import compress_features
from unifold.data.protein import PDB_CHAIN_IDS

from unittest import result
from unifold.config import model_config
from unifold.modules.alphafold import AlphaFold
from unifold.dataset import load_and_process, UnifoldDataset
from unicore.utils import (
    tensor_tree_map,
)
from unifold.symmetry import (
    UFSymmetry,
    uf_symmetry_config,
    assembly_from_prediction,
    load_and_process_symmetry,
)


MIN_SINGLE_SEQUENCE_LENGTH = 16
MAX_SINGLE_SEQUENCE_LENGTH = 1000
MAX_MULTIMER_LENGTH = 1000

logger = logging.getLogger(__name__)


def clean_and_validate_sequence(
        input_sequence: str, min_length: int, max_length: int) -> str:
    """Checks that the input sequence is ok and returns a clean version of it."""
    # Remove all whitespaces, tabs and end lines; upper-case.
    clean_sequence = input_sequence.translate(
        str.maketrans('', '', ' \n\t')).upper()
    aatypes = set(residue_constants.restypes)  # 20 standard aatypes.
    if not set(clean_sequence).issubset(aatypes):
        raise ValueError(
            f'Input sequence contains non-amino acid letters: '
            f'{set(clean_sequence) - aatypes}. AlphaFold only supports 20 standard '
            'amino acids as inputs.')
    if len(clean_sequence) < min_length:
        raise ValueError(
            f'Input sequence is too short: {len(clean_sequence)} amino acids, '
            f'while the minimum is {min_length}')
    if len(clean_sequence) > max_length:
        raise ValueError(
            f'Input sequence is too long: {len(clean_sequence)} amino acids, while '
            f'the maximum is {max_length}. You may be able to run it with the full '
            f'Uni-Fold system depending on your resources (system memory, '
            f'GPU memory).')
    return clean_sequence


def validate_input(
        input_sequences: Sequence[str],
        symmetry_group: str,
        min_length: int,
        max_length: int,
        max_multimer_length: int) -> Tuple[Sequence[str], bool]:
    """Validates and cleans input sequences and determines which model to use."""
    sequences = []

    for input_sequence in input_sequences:
        if input_sequence.strip():
            input_sequence = clean_and_validate_sequence(
                input_sequence=input_sequence,
                min_length=min_length,
                max_length=max_length)
            sequences.append(input_sequence)

    if symmetry_group != 'C1':
        if symmetry_group.startswith('C') and symmetry_group[1:].isnumeric():
            print(f'Using UF-Symmetry with group {symmetry_group}. If you do not '
                  f'want to use UF-Symmetry, please use `C1` and copy the AU '
                  f'sequences to the count in the assembly.')
            is_multimer = (len(sequences) > 1)
            return sequences, is_multimer, symmetry_group
        else:
            raise ValueError(f"UF-Symmetry does not support symmetry group "
                             f"{symmetry_group} currently. Cyclic groups (Cx) are "
                             f"supported only.")

    elif len(sequences) == 1:
        print('Using the single-chain model.')
        return sequences, False, None

    elif len(sequences) > 1:
        total_multimer_length = sum([len(seq) for seq in sequences])
        if total_multimer_length > max_multimer_length:
            raise ValueError(f'The total length of multimer sequences is too long: '
                             f'{total_multimer_length}, while the maximum is '
                             f'{max_multimer_length}. Please use the full AlphaFold '
                             f'system for long multimers.')
        print(f'Using the multimer model with {len(sequences)} sequences.')
        return sequences, True, None

    else:
        raise ValueError('No input amino acid sequence provided, please provide at '
                         'least one sequence.')


def add_hash(x, y):
    return x + "_" + hashlib.sha1(y.encode()).hexdigest()[:5]


TQDM_BAR_FORMAT = '{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed} remaining: {remaining}]'
DEFAULT_API_SERVER = "https://api.colabfold.com"


def run_mmseqs2(x, prefix, use_env=True,
                use_templates=False, use_pairing=False,
                host_url="https://api.colabfold.com") -> Tuple[List[str], List[str]]:
    submission_endpoint = "ticket/pair" if use_pairing else "ticket/msa"

    def submit(seqs, mode, N=101):
        n, query = N, ""
        for seq in seqs:
            query += f">{n}\n{seq}\n"
            n += 1

        res = requests.post(f'{host_url}/{submission_endpoint}', data={'q': query, 'mode': mode})
        try:
            out = res.json()
        except ValueError:
            logger.error(f"Server didn't reply with json: {res.text}")
            out = {"status": "ERROR"}
        return out

    def status(ID):
        res = requests.get(f'{host_url}/ticket/{ID}')
        try:
            out = res.json()
        except ValueError:
            logger.error(f"Server didn't reply with json: {res.text}")
            out = {"status": "ERROR"}
        return out

    def download(ID, path):
        res = requests.get(f'{host_url}/result/download/{ID}')
        with open(path, "wb") as out: out.write(res.content)

    # process input x
    seqs = [x] if isinstance(x, str) else x

    mode = "env"
    if use_pairing:
        mode = ""
        use_templates = False
        use_env = False

    # define path
    path = f"{prefix}"
    if not os.path.isdir(path): os.mkdir(path)

    # call mmseqs2 api
    tar_gz_file = f'{path}/out_{mode}.tar.gz'
    N, REDO = 101, True

    # deduplicate and keep track of order
    seqs_unique = []
    # TODO this might be slow for large sets
    [seqs_unique.append(x) for x in seqs if x not in seqs_unique]
    Ms = [N + seqs_unique.index(seq) for seq in seqs]
    # lets do it!
    if not os.path.isfile(tar_gz_file):
        TIME_ESTIMATE = 150 * len(seqs_unique)
        with tqdm(total=TIME_ESTIMATE, bar_format=TQDM_BAR_FORMAT) as pbar:
            while REDO:
                pbar.set_description("SUBMIT")

                # Resubmit job until it goes through
                out = submit(seqs_unique, mode, N)
                while out["status"] in ["UNKNOWN", "RATELIMIT"]:
                    sleep_time = 5 + random.randint(0, 5)
                    logger.error(f"Sleeping for {sleep_time}s. Reason: {out['status']}")
                    # resubmit
                    time.sleep(sleep_time)
                    out = submit(seqs_unique, mode, N)

                if out["status"] == "ERROR":
                    raise Exception(
                        f'MMseqs2 API is giving errors. Please confirm your input is a valid protein sequence. If error persists, please try again an hour later.')

                if out["status"] == "MAINTENANCE":
                    raise Exception(f'MMseqs2 API is undergoing maintenance. Please try again in a few minutes.')

                # wait for job to finish
                ID, TIME = out["id"], 0
                pbar.set_description(out["status"])
                while out["status"] in ["UNKNOWN", "RUNNING", "PENDING"]:
                    t = 5 + random.randint(0, 5)
                    logger.error(f"Sleeping for {t}s. Reason: {out['status']}")
                    time.sleep(t)
                    out = status(ID)
                    pbar.set_description(out["status"])
                    if out["status"] == "RUNNING":
                        TIME += t
                        pbar.update(n=t)

                if out["status"] == "COMPLETE":
                    if TIME < TIME_ESTIMATE:
                        pbar.update(n=(TIME_ESTIMATE - TIME))
                    REDO = False

                if out["status"] == "ERROR":
                    REDO = False
                    raise Exception(
                        f'MMseqs2 API is giving errors. Please confirm your input is a valid protein sequence. If error persists, please try again an hour later.')

            # Download results
            download(ID, tar_gz_file)

    # prep list of a3m files
    if use_pairing:
        a3m_files = [f"{path}/pair.a3m"]
    else:
        a3m_files = [f"{path}/uniref.a3m"]
        if use_env: a3m_files.append(f"{path}/bfd.mgnify30.metaeuk30.smag30.a3m")

    # extract a3m files
    if any(not os.path.isfile(a3m_file) for a3m_file in a3m_files):
        with tarfile.open(tar_gz_file) as tar_gz:
            tar_gz.extractall(path)

    # templates
    if use_templates:
        templates = {}

        for line in open(f"{path}/pdb70.m8", "r"):
            p = line.rstrip().split()
            M, pdb, qid, e_value = p[0], p[1], p[2], p[10]
            M = int(M)
            if M not in templates: templates[M] = []
            templates[M].append(pdb)

        template_paths = {}
        for k, TMPL in templates.items():
            TMPL_PATH = f"{prefix}/templates_{k}"
            if not os.path.isdir(TMPL_PATH):
                os.mkdir(TMPL_PATH)
                TMPL_LINE = ",".join(TMPL[:20])
                os.system(f"curl -s -L {host_url}/template/{TMPL_LINE} | tar xzf - -C {TMPL_PATH}/")
                os.system(f"cp {TMPL_PATH}/pdb70_a3m.ffindex {TMPL_PATH}/pdb70_cs219.ffindex")
                os.system(f"touch {TMPL_PATH}/pdb70_cs219.ffdata")
            template_paths[k] = TMPL_PATH

    # gather a3m lines
    a3m_lines = {}
    for a3m_file in a3m_files:
        update_M, M = True, None
        for line in open(a3m_file, "r"):
            if len(line) > 0:
                if "\x00" in line:
                    line = line.replace("\x00", "")
                    update_M = True
                if line.startswith(">") and update_M:
                    M = int(line[1:].rstrip())
                    update_M = False
                    if M not in a3m_lines: a3m_lines[M] = []
                a3m_lines[M].append(line)

    # return results

    a3m_lines = ["".join(a3m_lines[n]) for n in Ms]

    if use_templates:
        template_paths_ = []
        for n in Ms:
            if n not in template_paths:
                template_paths_.append(None)
                # print(f"{n-N}\tno_templates_found")
            else:
                template_paths_.append(template_paths[n])
        template_paths = template_paths_

    return (a3m_lines, template_paths) if use_templates else a3m_lines


def get_null_template(
        query_sequence: Union[List[str], str], num_temp: int = 1
) -> Dict[str, Any]:
    ln = (
        len(query_sequence)
        if isinstance(query_sequence, str)
        else sum(len(s) for s in query_sequence)
    )
    output_templates_sequence = "A" * ln
    output_confidence_scores = np.full(ln, 1.0)

    templates_all_atom_positions = np.zeros(
        (ln, templates.residue_constants.atom_type_num, 3)
    )
    templates_all_atom_masks = np.zeros((ln, templates.residue_constants.atom_type_num))
    templates_aatype = templates.residue_constants.sequence_to_onehot(
        output_templates_sequence, templates.residue_constants.HHBLITS_AA_TO_ID
    )
    template_features = {
        "template_all_atom_positions": np.tile(
            templates_all_atom_positions[None], [num_temp, 1, 1, 1]
        ),
        "template_all_atom_masks": np.tile(
            templates_all_atom_masks[None], [num_temp, 1, 1]
        ),
        "template_sequence": [f"none".encode()] * num_temp,
        "template_aatype": np.tile(np.array(templates_aatype)[None], [num_temp, 1, 1]),
        "template_domain_names": [f"none".encode()] * num_temp,
        "template_sum_probs": np.zeros([num_temp], dtype=np.float32),
    }
    return template_features


def get_template(
        a3m_lines: str, template_path: str, query_sequence: str
) -> Dict[str, Any]:
    template_featurizer = templates.HhsearchHitFeaturizer(
        mmcif_dir=template_path,
        max_template_date="2100-01-01",
        max_hits=20,
        kalign_binary_path="kalign",
        release_dates_path=None,
        obsolete_pdbs_path=None,
    )

    hhsearch_pdb70_runner = hhsearch.HHSearch(
        binary_path="hhsearch", databases=[f"{template_path}/pdb70"]
    )

    hhsearch_result = hhsearch_pdb70_runner.query(a3m_lines)
    hhsearch_hits = pipeline.parsers.parse_hhr(hhsearch_result)
    templates_result = template_featurizer.get_templates(
        query_sequence=query_sequence, hits=hhsearch_hits
    )
    return dict(templates_result.features)


def get_msa_and_templates(
        jobname: str,
        query_seqs_unique: Union[str, List[str]],
        result_dir: Path,
        msa_mode: str,
        use_templates: bool,
        homooligomers_num: int = 1,
        host_url: str = DEFAULT_API_SERVER,
) -> Tuple[
    Optional[List[str]], Optional[List[str]], List[str], List[int], List[Dict[str, Any]]
]:
    use_env = msa_mode == "MMseqs2"

    template_features = []
    if use_templates:
        a3m_lines_mmseqs2, template_paths = run_mmseqs2(
            query_seqs_unique,
            str(result_dir.joinpath(jobname)),
            use_env,
            use_templates=True,
            host_url=host_url,
        )
        if template_paths is None:
            logger.info("No template detected")
            for index in range(0, len(query_seqs_unique)):
                template_feature = get_null_template(query_seqs_unique[index])
                template_features.append(template_feature)
        else:
            for index in range(0, len(query_seqs_unique)):
                if template_paths[index] is not None:
                    template_feature = get_template(
                        a3m_lines_mmseqs2[index],
                        template_paths[index],
                        query_seqs_unique[index],
                    )
                    if len(template_feature["template_domain_names"]) == 0:
                        template_feature = get_null_template(query_seqs_unique[index])
                        logger.info(f"Sequence {index} found no templates")
                    else:
                        logger.info(
                            f"Sequence {index} found templates: {template_feature['template_domain_names'].astype(str).tolist()}"
                        )
                else:
                    template_feature = get_null_template(query_seqs_unique[index])
                    logger.info(f"Sequence {index} found no templates")

                template_features.append(template_feature)
    else:
        for index in range(0, len(query_seqs_unique)):
            template_feature = get_null_template(query_seqs_unique[index])
            template_features.append(template_feature)

    if msa_mode == "single_sequence":
        a3m_lines = []
        num = 101
        for i, seq in enumerate(query_seqs_unique):
            a3m_lines.append(">" + str(num + i) + "\n" + seq)
    else:
        # find normal a3ms
        a3m_lines = run_mmseqs2(
            query_seqs_unique,
            str(result_dir.joinpath(jobname)),
            use_env,
            use_pairing=False,
            host_url=host_url,
        )
    if len(query_seqs_unique) > 1:
        # find paired a3m if not a homooligomers
        paired_a3m_lines = run_mmseqs2(
            query_seqs_unique,
            str(result_dir.joinpath(jobname)),
            use_env,
            use_pairing=True,
            host_url=host_url,
        )
    else:
        num = 101
        paired_a3m_lines = []
        for i in range(0, homooligomers_num):
            paired_a3m_lines.append(
                ">" + str(num + i) + "\n" + query_seqs_unique[0] + "\n"
            )

    return (
        a3m_lines,
        paired_a3m_lines,
        template_features,
    )


def automatic_chunk_size(seq_len):
    if seq_len < 512:
        chunk_size = 256
    elif seq_len < 1024:
        chunk_size = 128
    elif seq_len < 2048:
        chunk_size = 32
    elif seq_len < 3072:
        chunk_size = 16
    else:
        chunk_size = 1
    return chunk_size


def load_feature_for_one_target(
        config, data_folder, seed=0, is_multimer=False, use_uniprot=False, symmetry_group=None,
):
    if not is_multimer:
        uniprot_msa_dir = None
        sequence_ids = ["A"]
        if use_uniprot:
            uniprot_msa_dir = data_folder

    else:
        uniprot_msa_dir = data_folder
        sequence_ids = open(os.path.join(data_folder, "chains.txt")).readline().split()

    if symmetry_group is None:
        batch, _ = load_and_process(
            config=config.data,
            mode="predict",
            seed=seed,
            batch_idx=None,
            data_idx=0,
            is_distillation=False,
            sequence_ids=sequence_ids,
            monomer_feature_dir=data_folder,
            uniprot_msa_dir=uniprot_msa_dir,
        )

    else:
        batch, _ = load_and_process_symmetry(
            config=config.data,
            mode="predict",
            seed=seed,
            batch_idx=None,
            data_idx=0,
            is_distillation=False,
            symmetry=symmetry_group,
            sequence_ids=sequence_ids,
            monomer_feature_dir=data_folder,
            uniprot_msa_dir=uniprot_msa_dir,
        )
    batch = UnifoldDataset.collater([batch])
    return batch


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='UniFold / ColabFold helper script')

    parser.add_argument('--input', required='True', help='Input fasta file.')
    parser.add_argument('--out_path', default='./prediction')
    parser.add_argument('--job_name', default='unifold_colab')
    parser.add_argument('--symmetry_group', default='C2')
    parser.add_argument('--use_templates', action='store_true')
    parser.add_argument('--msa_mode', default='MMseqs2', help='Allowed: \'MMseqs2\', \'single_sequence\'.')
    parser.add_argument('--weights_loc', default='/home/db/unifold/params/uf_symmetry.pt')
    parser.add_argument('--max_recycling_iters', default=3, type=int)
    parser.add_argument('--num_ensembles', default=2, type=int)
    parser.add_argument('--manual_seed', default=42, type=int)
    parser.add_argument('--times', default=3, type=int, help='It\'s not clear what this argument means, I suppose it\'s number of the obtained models.')

    args = parser.parse_args()

    output_dir_base = args.out_path
    os.makedirs(output_dir_base, exist_ok=True)

    inp = {str(entry.id): str(entry.seq) for entry in SeqIO.parse(args.input, 'fasta')}

    sequence_1 = ''  # @param {type:"string"}
    sequence_2 = ''  # @param {type:"string"}
    sequence_3 = ''  # @param {type:"string"}
    sequence_4 = ''  # @param {type:"string"}

    symmetry_group = args.symmetry_group
    use_templates = args.use_templates  # @param {type:"boolean"}
    msa_mode = args.msa_mode  # @param ["MMseqs2","single_sequence"]

    input_sequences = list(inp.values())

    jobname = args.job_name  # @param {type:"string"}

    basejobname = "".join(input_sequences)
    basejobname = re.sub(r'\W+', '', basejobname)
    target_id = jobname

    # Validate the input.
    sequences, is_multimer, symmetry_group = validate_input(
        input_sequences=input_sequences,
        symmetry_group=symmetry_group,
        min_length=MIN_SINGLE_SEQUENCE_LENGTH,
        max_length=MAX_SINGLE_SEQUENCE_LENGTH,
        max_multimer_length=MAX_MULTIMER_LENGTH)

    descriptions = ['> ' + target_id + ' seq' + str(ii) for ii in range(len(sequences))]

    if is_multimer:
        divide_multi_chains(target_id, output_dir_base, sequences, descriptions)

    s = []
    for des, seq in zip(descriptions, sequences):
        s += [des, seq]

    unique_sequences = []
    [unique_sequences.append(x) for x in sequences if x not in unique_sequences]

    if len(unique_sequences) == 1:
        homooligomers_num = len(sequences)
    else:
        homooligomers_num = 1

    # Is it needed?
    #with open(f"{jobname}.fasta", "w") as f:
        #f.write("\n".join(s))

    result_dir = Path(output_dir_base)
    output_dir = os.path.join(output_dir_base, target_id)

    (
      unpaired_msa,
      paired_msa,
      template_results,
    ) = get_msa_and_templates(
      target_id,
      unique_sequences,
      result_dir=result_dir,
      msa_mode=msa_mode,
      use_templates=use_templates,
      homooligomers_num = homooligomers_num
    )

    for idx, seq in enumerate(unique_sequences):
        chain_id = PDB_CHAIN_IDS[idx]
        sequence_features = pipeline.make_sequence_features(
                  sequence=seq, description=f'> {jobname} seq {chain_id}', num_res=len(seq)
              )
        monomer_msa = parsers.parse_a3m(unpaired_msa[idx])
        msa_features = pipeline.make_msa_features([monomer_msa])
        template_features = template_results[idx]
        feature_dict = {**sequence_features, **msa_features, **template_features}
        feature_dict = compress_features(feature_dict)
        features_output_path = os.path.join(
                output_dir, "{}.feature.pkl.gz".format(chain_id)
            )
        pickle.dump(
            feature_dict,
            gzip.GzipFile(features_output_path, "wb"),
            protocol=4
            )
        if is_multimer:
            multimer_msa = parsers.parse_a3m(paired_msa[idx])
            pair_features = pipeline.make_msa_features([multimer_msa])
            pair_feature_dict = compress_features(pair_features)
            uniprot_output_path = os.path.join(
                output_dir, "{}.uniprot.pkl.gz".format(chain_id)
            )
            pickle.dump(
                pair_feature_dict,
                gzip.GzipFile(uniprot_output_path, "wb"),
                protocol=4,
            )

    if symmetry_group is not None:
        model_name = "uf_symmetry"
        param_path = args.weights_loc

    # Rest will fail on our setup
    elif is_multimer:
        model_name = "multimer_ft"
        param_path = "./multimer.unifold.pt"
    else:
        model_name = "model_2_ft"
        param_path = "./monomer.unifold.pt"

    max_recycling_iters = args.max_recycling_iters  # @param {type:"integer"}
    num_ensembles = args.num_ensembles  # @param {type:"integer"}
    manual_seed = args.manual_seed  # @param {type:"integer"}
    times = args.times  # @param {type:"integer"}

    if symmetry_group is None:
        config = model_config(model_name)
    else:
        config = uf_symmetry_config()
    config.data.common.max_recycling_iters = max_recycling_iters
    config.globals.max_recycling_iters = max_recycling_iters
    config.data.predict.num_ensembles = num_ensembles

    # faster prediction with large chunk
    config.globals.chunk_size = 128
    model = AlphaFold(config) if symmetry_group is None else UFSymmetry(config)
    print("start to load params {}".format(param_path))
    state_dict = torch.load(param_path)["ema"]["params"]
    state_dict = {".".join(k.split(".")[1:]): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to("cuda:0")
    model.eval()
    model.inference_mode()

    # data path is based on target_name
    cur_param_path_postfix = os.path.split(param_path)[-1]

    print("start to predict {}".format(target_id))
    plddts = {}
    ptms = {}
    best_protein = None
    best_score = 0
    best_plddt = None
    best_pae = None

    for seed in range(times):
        cur_seed = hash((manual_seed, seed)) % 100000
        batch = load_feature_for_one_target(
            config,
            output_dir,
            cur_seed,
            is_multimer=is_multimer,
            use_uniprot=is_multimer,
            symmetry_group=symmetry_group,
        )
        seq_len = batch["aatype"].shape[-1]
        model.globals.chunk_size = automatic_chunk_size(seq_len)

        with torch.no_grad():
            batch = {
                k: torch.as_tensor(v, device="cuda:0")
                for k, v in batch.items()
            }
            shapes = {k: v.shape for k, v in batch.items()}
            #print(shapes) not meaningful for users
            t = time.perf_counter()
            out = model(batch)
            print(f"Inference time: {time.perf_counter() - t}")


        def to_float(x):
            if x.dtype == torch.bfloat16 or x.dtype == torch.half:
                return x.float()
            else:
                return x


        # Toss out the recycling dimensions --- we don't need them anymore
        batch = tensor_tree_map(lambda t: t[-1, 0, ...], batch)
        batch = tensor_tree_map(to_float, batch)
        out = tensor_tree_map(lambda t: t[0, ...], out)
        out = tensor_tree_map(to_float, out)
        batch = tensor_tree_map(lambda x: np.array(x.cpu()), batch)
        out = tensor_tree_map(lambda x: np.array(x.cpu()), out)

        plddt = out["plddt"]
        mean_plddt = np.mean(plddt)
        plddt_b_factors = np.repeat(
            plddt[..., None], residue_constants.atom_type_num, axis=-1
        )
        # TODO: , may need to reorder chains, based on entity_ids
        if symmetry_group is None:
            cur_protein = protein.from_prediction(
                features=batch, result=out, b_factors=plddt_b_factors
            )
        else:
            plddt_b_factors_assembly = np.concatenate(
                [plddt_b_factors for _ in range(batch["symmetry_opers"].shape[0])])
            cur_protein = assembly_from_prediction(
                result=out, b_factors=plddt_b_factors_assembly,
            )
        cur_save_name = (
            f"{cur_param_path_postfix}_{cur_seed}"
        )
        plddts[cur_save_name] = str(mean_plddt)
        if is_multimer and symmetry_group is None:
            ptms[cur_save_name] = str(np.mean(out["iptm+ptm"]))
        with open(os.path.join(output_dir, cur_save_name + '.pdb'), "w") as f:
            f.write(protein.to_pdb(cur_protein))

        if is_multimer and symmetry_group is None:
            mean_ptm = np.mean(out["iptm+ptm"])
            if mean_ptm > best_score:
                best_protein = cur_protein
                best_pae = out["predicted_aligned_error"]
                best_plddt = out["plddt"]
                best_score = mean_ptm
        else:
            if mean_plddt > best_score:
                best_protein = cur_protein
                best_plddt = out["plddt"]
                best_score = mean_plddt

    print("plddts", plddts)
    score_name = f"{model_name}_{cur_param_path_postfix}"
    plddt_fname = score_name + "_plddt.json"
    json.dump(plddts, open(os.path.join(output_dir, plddt_fname), "w"), indent=4)
    if ptms:
        print("ptms", ptms)
        ptm_fname = score_name + "_ptm.json"
        json.dump(ptms, open(os.path.join(output_dir, ptm_fname), "w"), indent=4)

