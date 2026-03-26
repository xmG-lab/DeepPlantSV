from __future__ import annotations

from collections import defaultdict, namedtuple
from itertools import product
from typing import Dict, Optional

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

from .constants import VOCAB

Coord = namedtuple("Coord", ["x", "y"])
CGRCoords = namedtuple("CGRCoords", ["N", "x", "y"])
DEFAULT_COORDS = dict(G=Coord(1, 1), C=Coord(-1, 1), A=Coord(-1, -1), T=Coord(1, -1))


class CGR:
    """Chaos Game Representation for DNA sequences."""

    def __init__(self, coords: Optional[Dict[str, tuple]] = None):
        self.nucleotide_coords = DEFAULT_COORDS if coords is None else coords
        self.cgr_coords = CGRCoords(0, 0, 0)

    def nucleotide_by_coords(self, x: int, y: int) -> str:
        filtered = dict(filter(lambda item: item[1] == Coord(x, y), self.nucleotide_coords.items()))
        return list(filtered.keys())[0]

    def forward(self, nucleotide: str) -> None:
        nuc_coord = self.nucleotide_coords.get(nucleotide.upper())
        if nuc_coord is None:
            return
        x = (self.cgr_coords.x + nuc_coord.x) / 2
        y = (self.cgr_coords.y + nuc_coord.y) / 2
        self.cgr_coords = CGRCoords(self.cgr_coords.N + 1, x, y)

    def backward(self) -> str:
        n_x, n_y = self.coords_current_nucleotide()
        nucleotide = self.nucleotide_by_coords(n_x, n_y)
        x = 2 * self.cgr_coords.x - n_x
        y = 2 * self.cgr_coords.y - n_y
        self.cgr_coords = CGRCoords(self.cgr_coords.N - 1, x, y)
        return nucleotide

    def coords_current_nucleotide(self) -> tuple[int, int]:
        x = 1 if self.cgr_coords.x > 0 else -1
        y = 1 if self.cgr_coords.y > 0 else -1
        return x, y

    def encode(self, sequence: str) -> CGRCoords:
        self.reset_coords()
        for nucleotide in sequence:
            self.forward(nucleotide)
        return self.cgr_coords

    def reset_coords(self) -> None:
        self.cgr_coords = CGRCoords(0, 0, 0)

    def decode(self, n: int, x: int, y: int) -> str:
        self.cgr_coords = CGRCoords(n, x, y)
        sequence = []
        while self.cgr_coords.N > 0:
            sequence.append(self.backward())
        return "".join(sequence[::-1])


class FCGR(CGR):
    """Frequency matrix CGR."""

    def __init__(self, k: int, bits: int = 8):
        super().__init__()
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")
        self.k = k
        self.kmers = ["".join(kmer) for kmer in product("ACGT", repeat=self.k)]
        self.kmer2pixel = self.kmer2pixel_position()
        self.freq_kmer = defaultdict(int)
        self.bits = bits
        self.max_color = 2 ** bits - 1

    def __call__(self, sequence: str) -> np.ndarray:
        if not isinstance(sequence, str) or len(sequence) < self.k:
            array_size = int(2 ** self.k)
            return np.zeros((array_size, array_size), dtype=np.float32)

        self.count_kmers(sequence.upper())
        array_size = int(2 ** self.k)
        fcgr = np.zeros((array_size, array_size), dtype=np.float32)
        if not self.freq_kmer:
            return fcgr

        for kmer, freq in self.freq_kmer.items():
            if kmer in self.kmer2pixel:
                pos_x, pos_y = self.kmer2pixel[kmer]
                fcgr[min(pos_x - 1, array_size - 1), min(pos_y - 1, array_size - 1)] = float(freq)
        return fcgr

    def count_kmer(self, kmer: str) -> None:
        if all(c in "ACGT" for c in kmer):
            self.freq_kmer[kmer] += 1

    def count_kmers(self, sequence: str) -> None:
        self.freq_kmer = defaultdict(int)
        last_j = len(sequence) - self.k + 1
        if last_j <= 0:
            return
        for i in range(last_j):
            self.count_kmer(sequence[i : i + self.k])

    def pixel_position(self, kmer: str) -> tuple[int, int]:
        coords = self.encode(kmer)
        x, y = coords.x, coords.y
        array_size = 2 ** self.k
        np_coords = np.array([(x + 1) / 2, (y + 1) / 2]) * array_size
        x_pos = min(array_size, max(1, int(np.ceil(np_coords[0]))))
        y_pos = min(array_size, max(1, int(np.ceil(np_coords[1]))))
        px = array_size - y_pos + 1
        py = x_pos
        return px, py

    def kmer2pixel_position(self) -> dict[str, tuple[int, int]]:
        return {kmer: self.pixel_position(kmer) for kmer in self.kmers}

    def array2img(self, array: np.ndarray) -> Image.Image:
        m, M = array.min(), array.max()
        img_rescaled = np.zeros_like(array) if M == m else (array - m) / (M - m)
        img_array = np.ceil(img_rescaled * self.max_color)
        dtype = np.uint8 if self.bits == 8 else np.uint16
        img_array = np.array(img_array, dtype=dtype)
        return Image.fromarray(img_array, "L")


def onehot_and_pad(sequence: str, max_len: int) -> np.ndarray:
    seq_idx = [VOCAB.get(base.upper(), -1) for base in sequence]
    if len(seq_idx) > max_len:
        seq_idx = seq_idx[:max_len]
    one_hot_matrix = np.zeros((len(seq_idx), 4), dtype=np.float32)
    for i, idx in enumerate(seq_idx):
        if idx != -1:
            one_hot_matrix[i, idx] = 1.0
    if len(seq_idx) < max_len:
        padding_needed = max_len - len(seq_idx)
        one_hot_matrix = np.concatenate([one_hot_matrix, np.zeros((padding_needed, 4), dtype=np.float32)], axis=0)
    return one_hot_matrix


def calculate_gc_content(sequence: str) -> float:
    seq_upper = sequence.upper()
    gc_count = 0
    total_bases = 0
    for base in seq_upper:
        if base in "GC":
            gc_count += 1
        if base in "ATGC":
            total_bases += 1
    return 0.0 if total_bases == 0 else gc_count / total_bases


def _validate_column(data: pd.DataFrame, col_name: str = "sequence") -> None:
    if col_name not in data.columns:
        raise KeyError(f"Missing required column: {col_name}. Available columns: {list(data.columns)}")


def generate_onehot_features(data: pd.DataFrame, max_len: int, col_name: str = "sequence") -> np.ndarray:
    _validate_column(data, col_name)
    return np.array([onehot_and_pad(seq if isinstance(seq, str) else "", max_len) for seq in tqdm(data[col_name], total=len(data), desc="OneHot")], dtype=np.float32)


def generate_fcgr_features(data: pd.DataFrame, k: int, col_name: str = "sequence") -> np.ndarray:
    _validate_column(data, col_name)
    fcgr_maker = FCGR(k=k)
    array_size = int(2 ** k)
    features = []
    for seq in tqdm(data[col_name], total=len(data), desc="FCGR"):
        if not isinstance(seq, str):
            features.append(np.zeros((array_size, array_size), dtype=np.float32))
        else:
            features.append(fcgr_maker(seq))
    return np.array(features, dtype=np.float32)


def generate_gc_features(data: pd.DataFrame, col_name: str = "sequence") -> np.ndarray:
    _validate_column(data, col_name)
    return np.array([calculate_gc_content(seq if isinstance(seq, str) else "") for seq in tqdm(data[col_name], total=len(data), desc="GC")], dtype=np.float32)
