#!/usr/bin/env python3

import os
import sys
import json
import struct
import hashlib
import logging

import numpy as np

from copy import deepcopy
from dataclasses import dataclass
from functools import cached_property
from argparse import ArgumentParser
from tqdm import tqdm


# Defaut block/chunk size for streaming reads
BLOCK_SIZE = 2**20


# Megatron-LM indexed dataset `.idx` file header
_INDEX_HEADER = b'MMIDIDX\x00\x00'


# Megatron-LM indexed dataset `.idx` file data type mapping
DTYPE_MAP = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float64,
    7: np.float32,
    8: np.uint16,
}


@dataclass
class IdxInfo:
    size: int
    dtype: str
    sequence_count: int
    total_tokens: int
    md5sum: str


@dataclass
class BinInfo:
    size: int
    md5sum: str


def bin_and_idx_paths(args):
    """Returns .bin and .idx paths, performing minimal sanity checking."""
    root, ext = os.path.splitext(args.bin)

    if ext == '.idx':
        raise ValueError(f'{args.bin} is .idx, expected .bin')
    elif ext != '.bin':
        logging.warning(f'{args.bin} has extension "{ext}", expected ".bin"')

    return args.bin, root+'.idx'


def cross_check(idxinfo, bininfo):
    """Sanity-check IdxInfo against BinInfo."""
    itemsize = np.dtype(idxinfo.dtype).itemsize

    # .bin is simple blob of total_tokens elements of size itemsize
    expected_size = idxinfo.total_tokens * itemsize
    if bininfo.size != expected_size:
        raise ValueError(f'.bin size {bininfo.size} does not match expected size {expected_size} ({idxinfo.total_tokens} total tokens, item size {itemsize})')


def info(args):
    """Implements megatrondata.py info"""
    bin_path, idx_path = bin_and_idx_paths(args)

    idxinfo = get_idx_info(idx_path)
    bininfo = get_bin_info(bin_path)

    cross_check(idxinfo, bininfo)

    data = {
        'idx' : idxinfo.__dict__,
        'bin' : bininfo.__dict__,
    }
    print(json.dumps(data, indent=2))


def verify_dicts(reference, data, label):
    data = deepcopy(data)
    for k, v in reference.items():
        if k not in data:
            raise ValueError(f'{label}: extra key in reference info: {k}')
        if data[k] != v:
            raise ValueError(f'{label}: {k} mismatch: {v} vs. {data[k]}')
        data.pop(k)
    if data:
        raise ValueError(f'missing key(s) from reference: {list(data.keys())}')


def verify(args):
    """Implements megatrondata.py verify"""
    with open(args.info) as f:
        reference = json.load(f)

    bin_path, idx_path = bin_and_idx_paths(args)

    idxinfo = get_idx_info(idx_path)
    verify_dicts(reference['idx'], idxinfo.__dict__, 'idx')

    bininfo = get_bin_info(bin_path)
    verify_dicts(reference['bin'], bininfo.__dict__, 'bin')

    print('OK')


def parse_args():
    ap = ArgumentParser()

    pp = ArgumentParser(add_help=False)    # shared args
    pp.add_argument('--quiet', action='store_true', help='less output')

    sp = ap.add_subparsers(dest='command', required=True)

    ip = sp.add_parser('info', parents=[pp])
    ip.add_argument('bin', help='Megatron .bin file')
    ip.set_defaults(func=info)

    vp = sp.add_parser('verify', parents=[pp])
    vp.add_argument('bin', help='Megatron .bin file')
    vp.add_argument('info', help='JSON file with info metadata')
    vp.set_defaults(func=verify)

    return ap.parse_args()


def md5sum(path, block_size=BLOCK_SIZE):
    """Return hashlib.file_digest(path, "md5").hexdigest(), displaying
    a progress bar while calculating the hash."""
    size = os.path.getsize(path)
    digestobj = hashlib.md5()
    buf = bytearray(block_size)
    view = memoryview(buf)
    with open(path, 'rb') as f:
        with tqdm(total=size, disable=tqdm.disable) as bar:
            while True:
                size = f.readinto(buf)
                if size == 0:
                    break
                digestobj.update(view[:size])
                bar.update(size)
    return digestobj.hexdigest()


class IndexReader(object):
    """Reads Megatron-LM `.idx` file header and provides some additional
    information.

    Mostly a reduced and simplified version of Megatron _IndexReader
    (from megatron/core/datasets/indexed_dataset.py) with some bonus
    functionality.

    Parameters
    ----------
    path : str
        Path to the `.idx` file.

    """
    def __init__(self, path: str):
        self.path = path

        with open(path, 'rb') as f:
            header = f.read(len(_INDEX_HEADER))
            assert header == _INDEX_HEADER, f'bad .idx header in {path}'

            version = struct.unpack('<Q', f.read(8))[0]
            assert version == 1, f'bad .idx version in {path}'

            code = struct.unpack('<B', f.read(1))[0]
            self.dtype = DTYPE_MAP[code]

            self.sequence_count = struct.unpack('<Q', f.read(8))[0]
            self.document_count = struct.unpack('<Q', f.read(8))[0]

            self.seq_len_offset = f.tell()
            seq_len_data_size = self.sequence_count * 4    # int32
            self.seq_ptr_offset = self.seq_len_offset + seq_len_data_size
            seq_ptr_data_size = self.sequence_count * 8    # int64
            self.doc_idx_offset = self.seq_ptr_offset + seq_ptr_data_size

    @cached_property
    def total_tokens(self) -> int:
        """Return total number of tokens."""

        with open(self.path, 'rb') as f:
            # read last sequence length value (number of tokens in last doc)
            seq_len_size = 4    # int32
            f.seek(self.seq_len_offset + (self.sequence_count-1)*seq_len_size)
            last_seq_len = struct.unpack('<i', f.read(4))[0]

            # read last sequence pointer value (where last doc starts)
            seq_ptr_size = 8    # int64
            f.seek(self.seq_ptr_offset + (self.sequence_count-1)*seq_ptr_size)
            last_seq_ptr = struct.unpack('<q', f.read(8))[0]

            assert last_seq_ptr % self.dtype().itemsize == 0   # sanity
            return last_seq_ptr//self.dtype().itemsize + last_seq_len

    def sanity_check(self, block_size=BLOCK_SIZE):
        """Check index data integrity, raise exception if there's an issue."""
        # streaming implementation to support large indices
        itemsize = self.dtype().itemsize
        with open(self.path, 'rb') as f:
            # check sequence_lengths and sequence_pointers
            current_pointer = 0
            for i in tqdm(
                    range(0, self.sequence_count, block_size),
                    total=(self.sequence_count+block_size-1)//block_size,
                    disable=tqdm.disable,
            ):
                n = min(block_size, self.sequence_count-i)

                # read sequence_lengths[i:i+n]
                f.seek(self.seq_len_offset + i*4)
                sequence_lengths = np.fromfile(f, np.int32, n)
                assert sequence_lengths.size == n, 'unexpected EOF'

                # read sequence_pointers[i:i+n]
                f.seek(self.seq_ptr_offset + i*8)
                sequence_pointers = np.fromfile(f, np.int64, n)
                assert sequence_pointers.size == n, 'unexpected EOF'

                if np.any(sequence_lengths < 0):
                    raise ValueError("negative value(s) in sequence_lengths")
                # TODO: consider warning but not failing on zeros
                if np.any(sequence_lengths == 0):
                    raise ValueError("zero value(s) in sequence_lengths")

                # sequence_pointers should be the cumulative sum of
                # sequence_lengths times itemsize (see Megatron
                # _IndexWriter._sequence_pointers())
                expected_pointers = current_pointer + np.cumsum(
                    np.concatenate(([0], sequence_lengths[:-1])),
                    dtype=np.int64
                ) * itemsize
                if not np.array_equal(sequence_pointers, expected_pointers):
                    raise ValueError('unexpected sequence_pointers value(s)')

                current_pointer += sequence_lengths.sum(dtype=np.int64) * itemsize
            assert current_pointer/itemsize == self.total_tokens

            # check document_indices
            f.seek(self.doc_idx_offset)
            for i in tqdm(
                    range(0, self.document_count, block_size),
                    total=(self.document_count+block_size-1)//block_size,
                    disable=tqdm.disable,
            ):
                n = min(block_size, self.document_count-i)
                document_indices = np.fromfile(f, np.int64, n)
                assert document_indices.size == n, 'unexpected EOF'

                # single sequence per document, so document indices
                # should be [0, 1, 2, ... n]
                expected_indices = np.arange(i, i+n, dtype=np.int32)
                if not np.array_equal(document_indices, expected_indices):
                    raise ValueError('unexpected document_indices value(s)')

            assert f.tell() == os.path.getsize(self.path), 'extra data'


def get_idx_info(path):
    """
    Return IdxInfo with information on given Megatron-LM `.idx` file.

    Parameters
    ----------
    path : str
        Path to the `.idx` file.

    Returns
    -------
    idxinfo: IdxInfo
        IdxInfo object with information extracted from `.idx` file.
    """
    logging.info(f'processing {path} ...')

    logging.info(f'checking size ...')
    size = os.path.getsize(path)
    assert size > 0, 'zero size .idx'

    logging.info(f'reading header ...')
    reader = IndexReader(path)

    logging.info('checking index data ...')
    reader.sanity_check()

    logging.info(f'taking checksum ...')
    checksum = md5sum(path)

    return IdxInfo(
        size=size,
        dtype=reader.dtype.__name__,
        sequence_count=reader.sequence_count,
        total_tokens=reader.total_tokens,
        md5sum=checksum,
    )


def get_bin_info(path):
    """
    Return BinInfo with information on given Megatron-LM `.bin` file.

    Parameters
    ----------
    path : str
        Path to the `.bin` file.

    Returns
    -------
    bininfo: BinInfo
        BinInfo object with information extracted from `.bin` file.
    """
    logging.info(f'processing {path} ...')

    logging.info(f'checking size ...')
    size = os.path.getsize(path)
    assert size > 0, 'zero size .bin'

    logging.info(f'taking checksum ...')
    checksum = md5sum(path)

    return BinInfo(
        size=size,
        md5sum=checksum,
    )


def main():
    args = parse_args()

    loglevel = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=loglevel)
    tqdm.disable = args.quiet    # hacky, sorry

    try:
        return args.func(args)
    except Exception as e:
        logging.error(e)
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
