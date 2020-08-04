
"""
Assignment 2 starter code
CSC148, Winter 2020
Instructors: Bogdan Simion, Michael Liut, and Paul Vrbik

This code is provided solely for the personal and private use of
students taking the CSC148 course at the University of Toronto.
Copying for purposes other than this use is expressly prohibited.
All forms of distribution of this code, whether as given or with
any changes, are expressly prohibited.

All of the files in this directory and all subdirectories are:
Copyright (c) 2020 Bogdan Simion, Michael Liut, Paul Vrbik, Dan Zingaro
"""
from __future__ import annotations
import time
from typing import Dict, Tuple
from utils import *
from huffman import HuffmanTree


# ====================
# Functions for compression


def build_frequency_dict(text: bytes) -> Dict[int, int]:
    """ Return a dictionary which maps each of the bytes in <text> to its
    frequency.

    >>> d = build_frequency_dict(bytes([65, 66, 67, 66]))
    >>> d == {65: 1, 66: 2, 67: 1}
    True
    """
    dic = {}
    for x in text:
        if x not in dic:
            dic.setdefault(x, 1)
        else:
            dic[x] += 1
    return dic

def transform(d: Dict[int, int]) -> List[tuple]:
    """ take in a dictionary of frequencies and return a sorted
    list of tuples (frequency = <value>, HuffmanTree(symbol = <key>))
    from highest to lowest frequency
    >>> transform({2: 6, 3: 4})
    [(6, HuffmanTree(2, None, None)), (4, HuffmanTree(3, None, None))]
    """
    r = []
    for x in d:
        a = HuffmanTree(x)
        r.append((d[x], a))
    #return r
    return sorted(r, key=lambda tup: tup[0], reverse=True)


def transform_v2(l: List[int]) -> List[tuple]:
    """ given a sorted (from smallest to largest frequency) list of tuples
     tuples are (<frequency>, <HT symbol>)
     return mutated list with <HT symbol> as a Huffmantree
     >>> l = [(104, 1), (101, 1), (119, 1), (114, 1), (100, 1), (111, 2), (108, 3)]
     >>> transform_v2(l)
     [(1, HuffmanTree(104, None, None)), (1, HuffmanTree(101, None, None)), (1, HuffmanTree(119, None, None)), (1, HuffmanTree(114, None, None)), (1, HuffmanTree(100, None, None)), (2, HuffmanTree(111, None, None)), (3, HuffmanTree(108, None, None))]
     """
    new_l = []
    for x in range(len(l)):
        a = HuffmanTree(l[x][0])
        new_l.append((l[x][1], a))
    return new_l


def format_tree(tree: HuffmanTree) -> HuffmanTree:
    """ return the correctly formatted version of a Huffman Tree"""
    a = tree
    if a.is_leaf():
        return a
    if a.left.is_leaf() and a.right.is_leaf():
        a.symbol = None
    else:
        if not(a.left.is_leaf()):
            format_tree(a.left)
        else:
            format_tree(a.left)
        if not(a.right.is_leaf()):
            format_tree(a.right)
        else:
            format_tree(a.right)
        if a.symbol is not None:
            a.symbol = None
    return a


def order(l: list, t: tuple) -> None:
    """mutate list to place tuple back into list at correct position"""
    freq = t[0]
    insert_number = 1
    freqs = []
    for x in range(len(l)):
        freqs.append(l[x][0])
        if freq <= l[x][0] and insert_number != 0:
            l.insert(x , t)
            insert_number -= 1
    if insert_number == 1 and len(freqs) == len(l):
        l.insert(len(l), t)


def build_huffman_tree(freq_dict: Dict[int, int]) -> HuffmanTree:
    """ Return the Huffman tree corresponding to the frequency dictionary
    <freq_dict>.

    Precondition: freq_dict is not empty.
    >>> freq = {2: 6}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(2, None, None)
    >>> t == result
    True
    >>> freq = {2: 6, 3: 4}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> t == result
    True
    >>> freq = {2: 6, 3: 4, 7: 5}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(2), \
                             HuffmanTree(None, HuffmanTree(3), HuffmanTree(7)))
    >>> t == result
    True
    >>> import random
    >>> symbol = random.randint(0,255)
    >>> freq = {symbol: 6}
    >>> t = build_huffman_tree(freq)
    >>> any_valid_byte_other_than_symbol = (symbol + 1) % 256
    >>> dummy_tree = HuffmanTree(any_valid_byte_other_than_symbol)
    >>> result = HuffmanTree(None, HuffmanTree(symbol), dummy_tree)
    >>> t.left == result.left or t.right == result.right
    False
    """
    sorted_d = sorted(freq_dict.items(), key=lambda x: x[1])
    lst = transform_v2(sorted_d)
    new_l = []
    last = []

    if len(lst) == 1 and new_l == []:
        return lst[0][1]
    if len(lst) == 2 and new_l == []:
        new_h = HuffmanTree(None, lst[0][1], lst[1][1])
        x = format_tree(new_h)
        return x

    while len(lst) > 1:
        new_f = lst[0][0] + lst[1][0]
        new_h = HuffmanTree(new_f, lst[0][1], lst[1][1])
        new_t = (new_f, new_h)
        new_l.append(new_t)
        lst.pop(0)
        lst.pop(0)
    if len(lst) == 1:
        order(new_l, lst[0])
        #new_l.append(lst[0])
        new_l = sorted(new_l, key=lambda tup: tup[0], reverse=False)
        while len(new_l) != 2:
            new_f = new_l[0][0] + new_l[1][0]
            new_h = HuffmanTree(new_f, new_l[0][1], new_l[1][1])
            new_t = (new_f, new_h)
            new_l.append(new_t)
            new_l.pop(0)
            new_l.pop(0)
        if len(new_l) == 2 and len(new_l) != 0:
            x = format_tree(new_l[0][1])
            y = format_tree(new_l[1][1])
            new_h = HuffmanTree(None, x, y)
            last.append(new_h)
    return last[0]


def merge(d1: dict, d2: dict)-> dict:
    """merge the values from the same key in two dictionaries """
    #d3 = [**d1, **d2]
    d3 = {**d1, **d2}
    #d3 = {d1, d2}
    for key, value in d3.items():
        if key in d1 and key in d2:
            d3[key] = d1[key]+ value
    return d3

def get_HTN(tree: HuffmanTree, d: dict) -> dict:
    """"return a dictionary of every HuffmanTree symbol set to None
    >>> tree = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> d = {}
    >>> get_HTN(tree, d)
    {3: None, 2: None}
    >>> tree = HuffmanTree(None, HuffmanTree(2), \
                             HuffmanTree(None, HuffmanTree(3), HuffmanTree(7)))
    >>> d = {}
    >>> get_HTN(tree, d)
    {2: None, 3: None, 7: None}
    """
    if tree.is_leaf():
        d.setdefault(tree.symbol)
    else:
        get_HTN(tree.left, d)
        get_HTN(tree.right, d)
    return d

def GC_helper(tree: HuffmanTree, counter: str, d: dict) -> Dict[int, str]:
    """ Return a dictionary which maps symbols from the Huffman tree <tree>
        to codes.
        with an additional <counter> parameter
    """
    if tree.is_leaf():
        d[tree.symbol] = counter
    #elif tree.left is not None or tree.right is not None:
    else:
        # if both left and right have sub HT
        if tree.left.is_leaf() and tree.right.is_leaf():
            counter_holder = counter
            if tree.left.is_leaf():
                #counter_holder = counter
                counter += '0'
                d[tree.left.symbol] = counter
                counter = counter_holder
            if tree.right.is_leaf():
                #counter_holder = counter
                counter += '1'
                d[tree.right.symbol] = counter
                counter = counter_holder
        if not(tree.left.is_leaf()) and not(tree.right.is_leaf()):
            counter_holder = counter
            if not (tree.left.is_leaf()):
                counter += '0'
                GC_helper(tree.left, counter, d)
                counter = counter_holder
            if not(tree.right.is_leaf()):
                counter += '1'
                GC_helper(tree.right, counter, d)
                counter = counter_holder
        # when left is a leaf and right has a subHT
        if not(tree.right.is_leaf()) and tree.left.is_leaf():
            counter_holder = counter
            counter += '0'
            GC_helper(tree.left, counter, d)
            counter = counter_holder
            counter += '1'
            GC_helper(tree.right, counter, d)

        # when right is a leaf and left has a subHT
        if not(tree.left.is_leaf()) and tree.right.is_leaf():
            GC_helper(tree.right, counter, d)
            counter += '0'
            GC_helper(tree.left, counter, d)
    return d


def get_codes(tree: HuffmanTree) -> Dict[int, str]:
    """ Return a dictionary which maps symbols from the Huffman tree <tree>
    to codes.

    >>> tree = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> d = get_codes(tree)
    >>> d == {3: "0", 2: "1"}
    True
    >>> tree = HuffmanTree(None, HuffmanTree(2), \
                             HuffmanTree(None, HuffmanTree(3), HuffmanTree(7)))
    >>> t = get_codes(tree)
    >>> result = {2: "0",3: "10",7: "11"}
    >>> t == result
    True
    >>> tree = build_huffman_tree(build_frequency_dict(b'helloworld'))
    >>> d = get_codes(tree)
    >>> result = {104: '000', 101: '001', 119: '010', 114: '011', 108: '10', 100: '110', 111: '111'}
    >>> d == result
    True
    """
    d = {}
    dl = GC_helper(tree.left, '0', get_HTN(tree.left, {}))
    dr = GC_helper(tree.right, '1', get_HTN(tree.right, {}))
    new = merge(dl, dr)
    d.update(new)
    return d


def num_nodes_helper(tree: HuffmanTree, n: int) -> None:
    """ Return mutated version of tree with internal nodes numbered in postorder
    with <n> being kept track of
    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(None, HuffmanTree(9), HuffmanTree(10))
    >>> tree = HuffmanTree(None, left, right)
    >>> num_nodes_helper(tree, 0)
    >>> tree.left.number
    0
    >>> tree.right.number
    1
    >>> tree.number
    2
    """
    if tree.left.is_leaf() and tree.right.is_leaf():
        tree.number = n
    else:
        if not(tree.left.is_leaf()):
            if not(tree.left.left.is_leaf()):
                num_nodes_helper(tree.left.left, n)
                n += 1
            if not(tree.left.right.is_leaf()):
                num_nodes_helper(tree.left.right, n)
                n += 1
            tree.left.number = n
            n += 1

        if not(tree.right.is_leaf()):
            if not(tree.right.left.is_leaf()):
                num_nodes_helper(tree.right.left, n)
                n += 1
            if not(tree.right.right.is_leaf()):
                num_nodes_helper(tree.right.right, n)
                n += 1
            tree.right.number = n
            n += 1
        tree.number = n


def number_nodes(tree: HuffmanTree) -> None:
    """ Number internal nodes in <tree> according to postorder traversal. The
    numbering starts at 0.
    >>> freq = build_frequency_dict(b"helloworld")
    >>> tree = build_huffman_tree(freq)
    >>> number_nodes(tree)

    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(None, HuffmanTree(9), HuffmanTree(10))
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> tree.left.number
    0
    >>> tree.right.number
    1
    >>> tree.number
    2
    """
    num_nodes_helper(tree, 0)


def avg_length(tree: HuffmanTree, freq_dict: Dict[int, int]) -> float:
    """ Return the average number of bits required per symbol, to compress the
    text made of the symbols and frequencies in <freq_dict>, using the Huffman
    tree <tree>.

    The average number of bits = the weighted sum of the length of each symbol
    (where the weights are given by the symbol's frequencies), divided by the
    total of all symbol frequencies.

    >>> freq = {3: 2, 2: 7, 9: 1}
    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(9)
    >>> tree = HuffmanTree(None, left, right)
    >>> avg_length(tree, freq)  # (2*2 + 7*2 + 1*1) / (2 + 7 + 1)
    1.9
    """
    average = 0
    total = 0
    library = get_codes(tree)
    for x in library:
        total += freq_dict[x]
        average += (freq_dict[x] * len(library[x]))
    return average/total


def fill(r: list) -> list:
    """ return r correctly formatted: every str in r has a length of 8"""
    want = 8
    for x in range(len(r)):
        b = ''
        counter = len(r[x])
        if want != counter:
            b = b.zfill(want - counter)
            r[x] += b
        else:
            pass
    return r


def compress_bytes(text: bytes, codes: Dict[int, str]) -> bytes:
    """ Return the compressed form of <text>, using the mapping from <codes>
    for each symbol.

    >>> d = {0: "0", 1: "10", 2: "11"}
    >>> text = bytes([1, 2, 1, 0])
    >>> result = compress_bytes(text, d)
    >>> result == bytes([184])
    True
    >>> [byte_to_bits(byte) for byte in result]
    ['10111000']
    >>> text = bytes([1, 2, 1, 0, 2])
    >>> result = compress_bytes(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111001', '10000000']
    """
    r = []
    counter = 0
    index = 0
    r.append("")
    for x in text:
        a = counter + len(codes[x])
        if a <= 8:
            r[index] += codes[x]
            counter += len(codes[x])
        elif counter == 8:
            counter = 0
            index += 1
            r.append("")
            a = len(codes[x])
            counter = len(codes[x])
            r[index] += codes[x]
        else:
            avail = 8 - counter
            r[index] += codes[x][:avail]
            index += 1
            r.append("")
            r[index] += codes[x][avail:]
            counter = len(r[index])
    r = fill(r)
    index = 0
    while index < len(r):
        a = bits_to_byte(r[index])
        r[index] = a
        index += 1
    return bytes(r)


def t_to_b_helper(tree: HuffmanTree, l: list) -> list:
    """ return
    >>> l = []
    >>> tree = build_huffman_tree(build_frequency_dict(b"helloworld"))
    >>> number_nodes(tree)
    >>> t_to_b_helper(tree, l)\
            #doctest: +NORMALIZE_WHITESPACE
    [0, 104, 0, 101, 0, 119, 0, 114, 1, 0, 1, 1, 0, 100, 0, 111, 0, 108, 1, 3, 1, 2, 1, 4]
    >>> l = []
    >>> tree = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> number_nodes(tree)
    >>> t_to_b_helper(tree, l)
    [0, 3, 0, 2]
    >>> l = []
    >>> left = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> right = HuffmanTree(5)
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> t_to_b_helper(tree, l)
    [0, 3, 0, 2, 1, 0, 0, 5]
    >>> l = []
    >>> tree = build_huffman_tree(build_frequency_dict(b"helloworld"))
    >>> number_nodes(tree)
    >>> t_to_b_helper(tree, l)\
            #doctest: +NORMALIZE_WHITESPACE
    [0, 104, 0, 101, 0, 119, 0, 114, 1, 0, 1, 1, 0, 100, 0, 111, 0, 108, 1, 3, 1, 2, 1, 4]
    """
    if tree.is_leaf():
        l += [0]
        l += [tree.symbol]
    else:
        if not(tree.left.is_leaf()) or not(tree.right.is_leaf()):
            if not(tree.left.is_leaf()) and not(tree.right.is_leaf()):
                t_to_b_helper(tree.left, l)
                t_to_b_helper(tree.right, l)
                l += [1]
                l += [tree.left.number]
                l += [1]
                l += [tree.right.number]
            elif not(tree.left.is_leaf()) and tree.right.is_leaf():
                t_to_b_helper(tree.left, l)
                l += [1]
                l += [tree.left.number]
                t_to_b_helper(tree.right, l)
            elif tree.left.is_leaf() and not(tree.right.is_leaf()):
                t_to_b_helper(tree.right, l)
                t_to_b_helper(tree.left, l)
                l += [1]
                l += [tree.right.number]

        else:
            t_to_b_helper(tree.left, l)
            t_to_b_helper(tree.right, l)
    return l


def tree_to_bytes(tree: HuffmanTree) -> bytes:
    """ Return a bytes representation of the Huffman tree <tree>.
    The representation should be based on the postorder traversal of the tree's
    internal nodes, starting from 0.

    Precondition: <tree> has its nodes numbered.

    >>> tree = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2]
    >>> left = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> right = HuffmanTree(5)
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2, 1, 0, 0, 5]
    >>> tree = build_huffman_tree(build_frequency_dict(b"helloworld"))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))\
            #doctest: +NORMALIZE_WHITESPACE
    [0, 104, 0, 101, 0, 119, 0, 114, 1, 0, 1, 1, 0, 100, 0, 111, 0, 108,\
    1, 3, 1, 2, 1, 4]
    """
    l = []
    r = t_to_b_helper(tree, l)
    return bytes(r)


def compress_file(in_file: str, out_file: str) -> None:
    """ Compress contents of the file <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    with open(in_file, "rb") as f1:
        text = f1.read()
    freq = build_frequency_dict(text)
    tree = build_huffman_tree(freq)
    codes = get_codes(tree)
    number_nodes(tree)
    print("Bits per symbol:", avg_length(tree, freq))
    result = (tree.num_nodes_to_bytes() + tree_to_bytes(tree) +
              int32_to_bytes(len(text)))
    result += compress_bytes(text, codes)
    with open(out_file, "wb") as f2:
        f2.write(result)


# ====================
# Functions for decompression

def generate_tree_general(node_lst: List[ReadNode],
                          root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes nothing about the order of the tree nodes in the list.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 1, 1, 0)]
    >>> generate_tree_general(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(10, None, None), \
HuffmanTree(12, None, None)), \
HuffmanTree(None, HuffmanTree(5, None, None), HuffmanTree(7, None, None)))
    """
    root = node_lst[root_index]
    HT = HuffmanTree(None, None, None)

    if root.l_type == 0 and root.r_type == 0:
        HT.left = HuffmanTree(root.l_data)
        HT.right = HuffmanTree(root.r_data)
    if root.l_type == 1 and root.r_type == 1:
        HT.left = generate_tree_general(node_lst, root.l_data)
        HT.right = generate_tree_general(node_lst, root.r_data)
    if root.l_type == 0 and root.r_type == 1:
        HT.left = HuffmanTree(root.l_data)
        HT.right = generate_tree_general(node_lst, root.r_data)
    if root.l_type == 1 and root.r_type == 0:
        HT.left = generate_tree_general(node_lst, root.l_data)
        HT.right = HuffmanTree(root.r_data)
    return HT


def build_HT(node_lst: List[ReadNode], index: int, root_index: int) -> HuffmanTree:
    """ return a HuffmanTree for function generate_tree_postorder """
    root = node_lst[root_index]
    RN = node_lst[index]
    HT = HuffmanTree(None, None, None)

    if RN.l_type == 0 and RN.r_type == 0:
        HT.left = HuffmanTree(RN.l_data)
        HT.right = HuffmanTree(RN.r_data)
    if RN.l_type == 1 and RN.r_type == 1:
        HT.left = generate_tree_general(node_lst, root.l_data)
        HT.right = generate_tree_general(node_lst, root.r_data)
    if RN.l_type == 0 and RN.r_type == 1:
        HT.left = HuffmanTree(root.l_data)
        HT.right = generate_tree_general(node_lst, root.r_data)
    if RN.l_type == 1 and RN.r_type == 0:
        HT.left = generate_tree_general(node_lst, root.l_data)
        HT.right = HuffmanTree(root.r_data)
    return HT


def generate_tree_postorder(node_lst: List[ReadNode],
                            root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes that the list represents a tree in postorder.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 0, 1, 0)]
    >>> generate_tree_postorder(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(5, None, None), \
HuffmanTree(7, None, None)), \
HuffmanTree(None, HuffmanTree(10, None, None), HuffmanTree(12, None, None)))
    """
    root = node_lst[root_index]
    temp = []
    left = 1
    right = 1
    HT = HuffmanTree(None, None, None)
    while left == 1:
        if root.l_type == 0:
            HT.left = HuffmanTree(root.l_data)
            left -= 1
        if root.l_type == 1:
            index = root.l_data
            HT.left = generate_tree_general(node_lst, index)
            #HT.left = build_HT(node_lst, index, root_index)
            temp.append(node_lst[index])
            node_lst.pop(index)
            root_index = node_lst.index(root)
            left -=1
    while right == 1:
        if root.r_type == 0:
            HT.right = HuffmanTree(root.r_data)
            right -= 1
        if root.r_type == 1:
            index = root.r_data
            HT.right = generate_tree_general(node_lst, index)
            #HT.right = build_HT(node_lst, index, root_index)
            temp.append(node_lst[index])
            node_lst.pop(index)
            root_index = node_lst.index(root)
            right -=1
    return HT


def decompress_bytes(tree: HuffmanTree, text: bytes, size: int) -> bytes:
    """ Use Huffman tree <tree> to decompress <size> bytes from <text>.

    >>> tree = build_huffman_tree(build_frequency_dict(b'helloworld'))
    >>> number_nodes(tree)
    >>> decompress_bytes(tree, compress_bytes(b'helloworld', get_codes(tree)), len(b'helloworld'))
    b'helloworld'
    """
    lst = []
    for x in text:
        lst.extend(byte_to_bits(x))
    symbols = []
    index = 0
    current = tree
    while index < len(lst) and len(symbols) != size:
        if current.is_leaf():
            symbols.append(current.symbol)
            current = tree
            index -=1
        else:
            if lst[index] == '0':
                current = current.left
            elif lst[index] == '1':
                current = current.right
        index += 1

    return bytes(symbols)


def decompress_file(in_file: str, out_file: str) -> None:
    """ Decompress contents of <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    with open(in_file, "rb") as f:
        num_nodes = f.read(1)[0]
        buf = f.read(num_nodes * 4)
        node_lst = bytes_to_nodes(buf)
        # use generate_tree_general or generate_tree_postorder here
        tree = generate_tree_general(node_lst, num_nodes - 1)
        size = bytes_to_int(f.read(4))
        with open(out_file, "wb") as g:
            text = f.read()
            g.write(decompress_bytes(tree, text, size))


# ====================
# Other functions

def freq_to_symbols(d: dict)-> dict:
    """ idont have time for this"""
    dic = {}
    for x in d:
        if d[x] in dic:
            dic[d[x]].append(x)
        else:
            dic[d[x]] = [x]
    for y in dic:
        dic[y].sort()
    return dic


def map_depths(tree: HuffmanTree, d_num: dict, d_list: dict) -> Tuple[dict,dict]:
    """ map out the values associated/ located at a certain depth and
    the number of items at that level
    key = <depth> and value = Tuple(# of items, [<item found at that depth>])
    """
    item_to_path = get_codes(tree)
    path_to_item = freq_to_symbols(item_to_path)
    d = {}
    for x in path_to_item:
        key = len(x)
        if key not in d:
            d[key] = []
        else:
            pass
    for l in path_to_item:
        a = len(l)
        if a in d:
            d[a].extend(path_to_item[l])
    for num in d:
        number = len(d[num])
        lst = d[num]
        d_num[num] =  number
        d_list[num] = lst
    return (d_num, d_list)

def improve_tree_helper(tree: HuffmanTree, dl: dict, inverse_dic: dict, depth: int):
    """ helper """
    if tree.symbol is not None:
        lst_of_items_at_level = dl[depth]
        an_item = lst_of_items_at_level.pop()
        tree.symbol = an_item
        #tree.symbol = inverse_dic[an_item].pop(0)

    else:
        for x in [tree.left, tree.right]:
            improve_tree_helper(x, dl, inverse_dic, depth + 1)



def improve_tree(tree: HuffmanTree, freq_dict: Dict[int, int]) -> None:
    """ Improve the tree <tree> as much as possible, without changing its shape,
    by swapping nodes. The improvements are with respect to the dictionary of
    symbol frequencies <freq_dict>.

    >>> left = HuffmanTree(None, HuffmanTree(99, None, None), \
    HuffmanTree(100, None, None))
    >>> right = HuffmanTree(None, HuffmanTree(101, None, None), \
    HuffmanTree(None, HuffmanTree(97, None, None), HuffmanTree(98, None, None)))
    >>> tree = HuffmanTree(None, left, right)
    >>> freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    >>> avg_length(tree, freq)
    2.49
    >>> improve_tree(tree, freq)
    >>> avg_length(tree, freq)
    2.31
    """
    # got lst of items in HT
    dic = get_HTN(tree, {})
    lst = []
    for x in dic:
        lst.append(x)

    # dictionary with frequency: symbol
    dic_v_2_k = freq_to_symbols(freq_dict)

    # dictionaries
    depths = map_depths(tree, {}, {})
    dn = depths[0]
    dl = depths[1]

    improve_tree_helper(tree, dl, dic_v_2_k, 0)










if __name__ == "__main__":
    import doctest
    doctest.testmod()

    import python_ta
    python_ta.check_all(config={
        'allowed-io': ['compress_file', 'decompress_file'],
        'allowed-import-modules': [
            'python_ta', 'doctest', 'typing', '__future__',
            'time', 'utils', 'huffman', 'random'
        ],
        'disable': ['W0401']
    })

    mode = input("Press c to compress, d to decompress, or other key to exit: ")
    if mode == "c":
        fname = input("File to compress: ")
        start = time.time()
        compress_file(fname, fname + ".huf")
        print("Compressed {} in {} seconds."
              .format(fname, time.time() - start))
    elif mode == "d":
        fname = input("File to decompress: ")
        start = time.time()
        decompress_file(fname, fname + ".orig")
        print("Decompressed {} in {} seconds."
              .format(fname, time.time() - start))
