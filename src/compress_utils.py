#!/usr/bin/env python3
# compress_utils.py - auxiliar simples para chamar utils.save_compressed_versions se necessário.
from typing import List
import utils, os

def compress_folder(src: str, dst: str, qualities: List[int] = [95,75,50,25]):
    return utils.save_compressed_versions(src, dst, qualities)
