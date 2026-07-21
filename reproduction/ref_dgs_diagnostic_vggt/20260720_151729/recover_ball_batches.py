#!/usr/bin/env python3
import argparse
import hashlib
import json
import pickletools
import struct
from collections import defaultdict
from pathlib import Path


FRAME_HEIGHT = 518
FRAME_WIDTH = 518
FRAME_AREA = FRAME_HEIGHT * FRAME_WIDTH
PAYLOAD_HASH_BYTES = 8_000_000
LOCAL_HEADER = b"PK\x03\x04"


def _local_record_data_start(blob: bytes, record_index: int) -> int:
    offsets = []
    cursor = 0
    while True:
        offset = blob.find(LOCAL_HEADER, cursor)
        if offset < 0:
            break
        offsets.append(offset)
        cursor = offset + len(LOCAL_HEADER)
    if len(offsets) <= record_index:
        raise ValueError(f"missing ZIP local record {record_index}")
    offset = offsets[record_index]
    name_length, extra_length = struct.unpack_from("<HH", blob, offset + 26)
    return offset + 30 + name_length + extra_length


def _storage_offset(blob: bytes) -> int:
    pickle_start = _local_record_data_start(blob, 0)
    operations = list(pickletools.genops(blob[pickle_start:4096]))
    offsets = [
        operations[index + 1][1]
        for index, operation in enumerate(operations[:-1])
        if operation[0].name == "BINPERSID"
    ]
    if len(offsets) != 2 or offsets[0] != offsets[1]:
        raise ValueError(f"expected two equal tensor storage offsets, got {offsets}")
    return int(offsets[0])


def _frame_number(path: Path) -> int:
    stem = path.stem
    if not stem.startswith("r_"):
        raise ValueError(f"unexpected prior filename: {path.name}")
    return int(stem[2:])


def recover_batch_manifest(depth_dir: Path) -> list[list[dict]]:
    depth_dir = Path(depth_dir)
    files = sorted(depth_dir.glob("r_*.pth"), key=_frame_number)
    if not files:
        raise ValueError(f"no depth priors found in {depth_dir}")

    grouped = defaultdict(list)
    for path in files:
        blob = path.read_bytes()
        storage_offset = _storage_offset(blob)
        if storage_offset % FRAME_AREA:
            raise ValueError(f"unaligned storage offset in {path}: {storage_offset}")
        slot = storage_offset // FRAME_AREA
        if not 0 <= slot < 20:
            raise ValueError(f"slot outside 20-view batch in {path}: {slot}")
        tensor_start = _local_record_data_start(blob, 2)
        tensor_prefix = blob[tensor_start : tensor_start + PAYLOAD_HASH_BYTES]
        if len(tensor_prefix) != PAYLOAD_HASH_BYTES:
            raise ValueError(f"insufficient intact storage prefix in {path}")
        group_hash = hashlib.sha256(tensor_prefix).hexdigest()
        grouped[group_hash].append(
            {
                "image_name": path.stem,
                "frame_number": _frame_number(path),
                "slot": slot,
                "official_prior_path": str(path),
            }
        )

    batches = []
    for records in grouped.values():
        records.sort(key=lambda item: item["slot"])
        if len(records) != 20 or [item["slot"] for item in records] != list(range(20)):
            raise ValueError("recovered group is not one complete 20-view batch")
        batches.append(records)
    batches.sort(key=lambda batch: min(item["frame_number"] for item in batch))
    return batches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-depth", required=True, type=Path)
    parser.add_argument("--test-depth", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = {
        "protocol": "recovered_from_intact_official_pickle_metadata",
        "frame_shape": [FRAME_HEIGHT, FRAME_WIDTH],
        "views_per_batch": 20,
        "train_batches": recover_batch_manifest(args.train_depth),
        "test_batches": recover_batch_manifest(args.test_depth),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"train_batches": len(payload["train_batches"]), "test_batches": len(payload["test_batches"])}))


if __name__ == "__main__":
    main()
