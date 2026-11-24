"""Restore the original replay inputs into a local copy of this code.

Usage: python fetch_replay_inputs.py --dest /path/to/cost-of-balance
Use --from-checkout to copy from a local copy of the original archive instead.
The historical inference workers also require the study backend and weights.
"""

import argparse
from hashlib import sha1
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import quote
from urllib.request import urlopen


ARCHIVE_URL = (
    "https://raw.githubusercontent.com/ricardofitas/ai-epistemics-hackathon/"
    "ebb5ec03fa85e585127b8a31af3e8aeafc4b7487/"
)
INPUTS = (
    ('certificates/adjacency-certificate.json',
     'campaign/option-order-design-20261122/icml-draft/certificates/adjacency-certificate.json',
     'd008cfcc995f52789e502ba590218449dd60bee3'),
    ('certificates/adjacency-witnesses.json',
     'campaign/option-order-design-20261122/icml-draft/certificates/adjacency-witnesses.json',
     '8630190fab0e879ebd6048ea39caaa049eace316'),
    ('certificates/binary-adjacency-certificate.json',
     'campaign/option-order-design-20261122/icml-draft/certificates/binary-adjacency-certificate.json',
     '3e609f70740423e732dd8710c39889b1c605dcf3'),
    ('certificates/five-correctness-witnesses.json',
     'campaign/option-order-design-20261122/icml-draft/certificates/five-correctness-witnesses.json',
     '50f2dd8c4944c7db4e10cf945bfab1907b0b6fba'),
    ('certificates/four-budget-certificate.json',
     'campaign/option-order-design-20261122/icml-draft/certificates/four-budget-certificate.json',
     '9d340eb8d07e156ab562ff792165ff500b6dc7d3'),
    ('certificates/four-budget-templates.json',
     'campaign/option-order-design-20261122/icml-draft/certificates/four-budget-templates.json',
     'c08dd307d9547268bb68c38b72b224142673a81f'),
    ('certificates/position-certificate.json',
     'campaign/option-order-design-20261122/icml-draft/certificates/position-certificate.json',
     'bad5fe037b06572c64e1c798401112e71666d434'),
    ('certificates/position-family-certificate.json',
     'campaign/option-order-design-20261122/icml-draft/certificates/position-family-certificate.json',
     '72f3e06c15de04b64ffb10036d84310cfd623874'),
    ('certificates/position-templates.json',
     'campaign/option-order-design-20261122/icml-draft/certificates/position-templates.json',
     '598edd38969b9c87d16854971441d9586720eb9a'),
    ('certificates/real-adjacency-certificate.json',
     'campaign/option-order-design-20261122/icml-draft/certificates/real-adjacency-certificate.json',
     '0e964857b7e388f0bea39867783ca01863cdbb64'),
    ('certificates/solutions.json',
     'campaign/option-order-design-20261122/icml-draft/certificates/solutions.json',
     '9ad7f0abdf43cc93f6e103d6ddb088974e70d139'),
    ('experiments/demo-order/option-costs.json',
     'campaign/option-order-design-20261122/icml-draft/experiments/demo-order/option-costs.json',
     'edca10f18ac58dd832ef1fc67e0149f06e9eebc7'),
    ('experiments/demo-order/receipt.json',
     'campaign/option-order-design-20261122/icml-draft/experiments/demo-order/receipt.json',
     '2ee94cd783c8771f7eb31c7e516363d95c431b27'),
    ('experiments/demo-order/responses.json',
     'campaign/option-order-design-20261122/icml-draft/experiments/demo-order/responses.json',
     'dbc62786b2309b21b26b417a6d766ae30dd43a34'),
    ('experiments/demo-order/results.json',
     'campaign/option-order-design-20261122/icml-draft/experiments/demo-order/results.json',
     '60cda0a949d938d68627834ae9d0978a515e9809'),
    ('experiments/eight-calls/profiles.json.gz',
     'campaign/option-order-design-20261122/icml-draft/experiments/eight-calls/profiles.json.gz',
     '9b2b3aa684d930ba1586d6b6c188dcb833278211'),
    ('experiments/eight-calls/results.json',
     'campaign/option-order-design-20261122/icml-draft/experiments/eight-calls/results.json',
     '0f20668a2ba0a9ede9085e973a02cdf0b3958c1c'),
    ('experiments/eight-calls/source-receipts.json',
     'campaign/option-order-design-20261122/icml-draft/experiments/eight-calls/source-receipts.json',
     'dfce2c3b35e7a0cf04fb4e6872c2d32144e8cbc3'),
    ('experiments/low-budget-selection/admission.json',
     'campaign/option-order-design-20261122/icml-draft/experiments/low-budget-selection/admission.json',
     '6deff46e5fb32fe38ffb9d7c6aaf07cbf4a7f360'),
    ('experiments/low-budget-selection/decisions.json',
     'campaign/option-order-design-20261122/icml-draft/experiments/low-budget-selection/decisions.json',
     '5e200dd956439a8d3d50b56e6aaf38b97952c1d1'),
    ('experiments/low-budget-selection/development.json',
     'campaign/option-order-design-20261122/icml-draft/experiments/low-budget-selection/development.json',
     '929305b34d58ac778ff17bf32aa3b78583c3b1bc'),
    ('experiments/low-budget-selection/join-audit.json',
     'campaign/option-order-design-20261122/icml-draft/experiments/low-budget-selection/join-audit.json',
     '15c29211ca72939989fc6d73186283b50e2f74e9'),
    ('experiments/low-budget-selection/pilot-observations.json',
     'campaign/option-order-design-20261122/icml-draft/experiments/low-budget-selection/pilot-observations.json',
     '798449e53c5361b08e2804afb65b60ccb5df130a'),
    ('experiments/low-budget-selection/profiles.json.gz',
     'campaign/option-order-design-20261122/icml-draft/experiments/low-budget-selection/profiles.json.gz',
     '88958f50a6ee026653b2dd9d82bde8c9957a189e'),
    ('experiments/low-budget-selection/registered-design.json',
     'campaign/option-order-design-20261122/icml-draft/experiments/low-budget-selection/registered-design.json',
     '9522b1ec624775803327b7048288dce46a95e533'),
    ('experiments/low-budget-selection/results.json',
     'campaign/option-order-design-20261122/icml-draft/experiments/low-budget-selection/results.json',
     '276537f8cd5dd850b0dc176303c55fc77a641ce5'),
    ('experiments/low-budget-selection/run-receipt.json',
     'campaign/option-order-design-20261122/icml-draft/experiments/low-budget-selection/run-receipt.json',
     '679956842f8810b4ef5c8b6eb7c1861486f4943a'),
    ('experiments/low-budget-selection/source-pins.json',
     'campaign/option-order-design-20261122/icml-draft/experiments/low-budget-selection/source-pins.json',
     '7ae784165e232be15f7b7b2fc106096452044def'),
    ('risk-components.json',
     'campaign/option-order-design-20261122/icml-draft/risk-components.json',
     '7519f1fc5a3dac7dd8dfee3402854983075bebe8'),
    ('evidence/rig-20261123/3b-gpu-qualification.json',
     'evidence/rig-20261123/3b-gpu-qualification.json',
     '1943f22e9f5bc2e7f2e88ced70efb00a76c281c7'),
    ('evidence/rig-20261123/aggregate-results.json',
     'evidence/rig-20261123/aggregate-results.json',
     '88c8b5668cde833f741c83c2b8d6820c33dae684'),
    ('evidence/rig-20261123/extended-results.json',
     'evidence/rig-20261123/extended-results.json',
     '44339481c0012447efd3e768e17f4c60882f0dd7'),
    ('evidence/rig-20261123/five-orders-commonsenseqa/PROTOCOL.md',
     'evidence/rig-20261123/five-orders-commonsenseqa/PROTOCOL.md',
     '0bc31a365111d9eb10210e8ff4aaac3ba296a76f'),
    ('evidence/rig-20261123/five-orders-commonsenseqa/binary-adjacency-certificate.json',
     'evidence/rig-20261123/five-orders-commonsenseqa/binary-adjacency-certificate.json',
     '3e609f70740423e732dd8710c39889b1c605dcf3'),
    ('evidence/rig-20261123/five-orders-commonsenseqa/commitment.json',
     'evidence/rig-20261123/five-orders-commonsenseqa/commitment.json',
     '0921f7df7fdab885a8a955034ec647aeb768c2dd'),
    ('evidence/rig-20261123/five-orders-commonsenseqa/correctness-masks.json',
     'evidence/rig-20261123/five-orders-commonsenseqa/correctness-masks.json',
     '72e98b6bf6e52c0873ec0610da5fa57f5247b8d8'),
    ('evidence/rig-20261123/five-orders-commonsenseqa/empirical-replay.json',
     'evidence/rig-20261123/five-orders-commonsenseqa/empirical-replay.json',
     '44be94869386768b26f8f8c078642a759fd02a5d'),
    ('evidence/rig-20261123/five-orders-commonsenseqa/frozen.json',
     'evidence/rig-20261123/five-orders-commonsenseqa/frozen.json',
     'e025721f63bfcb7c1c74a3e7da6c25e762e201f8'),
    ('evidence/rig-20261123/five-orders-commonsenseqa/position-certificate.json',
     'evidence/rig-20261123/five-orders-commonsenseqa/position-certificate.json',
     'bad5fe037b06572c64e1c798401112e71666d434'),
    ('evidence/rig-20261123/five-orders-commonsenseqa/real-adjacency-certificate.json',
     'evidence/rig-20261123/five-orders-commonsenseqa/real-adjacency-certificate.json',
     '0e964857b7e388f0bea39867783ca01863cdbb64'),
    ('evidence/rig-20261123/five-orders-commonsenseqa/results.json',
     'evidence/rig-20261123/five-orders-commonsenseqa/results.json',
     '46c271ab0f60a32c15319ae206863da5ebef5f3c'),
    ('evidence/rig-20261123/five-orders-commonsenseqa/solutions.json',
     'evidence/rig-20261123/five-orders-commonsenseqa/solutions.json',
     '9ad7f0abdf43cc93f6e103d6ddb088974e70d139'),
    ('evidence/rig-20261123/four-order-development-screen.json',
     'evidence/rig-20261123/four-order-development-screen.json',
     '8cd127afe34425e5909db4f09a28f9d13bf92879'),
    ('evidence/rig-20261123/four-orders-hellaswag-3b/correctness-masks.json',
     'evidence/rig-20261123/four-orders-hellaswag-3b/correctness-masks.json',
     'a95c1182a404273f032f5bb1c71b558e0e6bed8b'),
    ('evidence/rig-20261123/four-orders-hellaswag-3b/public-commitments.json',
     'evidence/rig-20261123/four-orders-hellaswag-3b/public-commitments.json',
     '6de32b8c20605d647ee45be0daabaa69326081be'),
    ('evidence/rig-20261123/four-orders-hellaswag-3b/results.json',
     'evidence/rig-20261123/four-orders-hellaswag-3b/results.json',
     '1544308d6b4055201b35aec600b2103c348c8171'),
    ('evidence/rig-20261123/four-orders-near-tie/preregistration.json',
     'evidence/rig-20261123/four-orders-near-tie/preregistration.json',
     '1e7d01b5859c3ed0e30f6e91b079067cdf83f40c'),
    ('evidence/rig-20261123/four-orders-near-tie/results.json',
     'evidence/rig-20261123/four-orders-near-tie/results.json',
     '91d783722e4d77f0a389df820e095c6c36fd76ce'),
    ('evidence/rig-20261123/four-orders-prospective-screen/results.json',
     'evidence/rig-20261123/four-orders-prospective-screen/results.json',
     'ce6e01a072861b724c047109407c776ad4207f5b'),
)


def blob_hash(contents):
    prefix = f"blob {len(contents)}\0".encode("ascii")
    return sha1(prefix + contents).hexdigest()


def restore(destination, checkout=None):
    for target, archive_path, expected_sha in INPUTS:
        path = destination / target
        if path.exists():
            if blob_hash(path.read_bytes()) != expected_sha:
                raise ValueError(f"Existing file differs from the study snapshot: {path}")
            continue
        if checkout is None:
            with urlopen(ARCHIVE_URL + quote(archive_path), timeout=60) as response:
                contents = response.read()
        else:
            contents = (checkout / archive_path).read_bytes()
        if blob_hash(contents) != expected_sha:
            raise ValueError(f"Unexpected archive hash for {archive_path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(contents)
        try:
            path.hardlink_to(temporary_path)
        finally:
            temporary_path.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--from-checkout", type=Path)
    args = parser.parse_args()
    restore(args.dest.resolve(), args.from_checkout)
