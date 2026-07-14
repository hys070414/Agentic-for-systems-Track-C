from __future__ import annotations

from typing import Dict, List, Set, Tuple

from ir import AECInstruction


OP_ADD = 0x0001
OP_SUB = 0x0002
OP_MUL = 0x0003
OP_MAD = 0x0004
OP_FMA = 0x0005
OP_AND = 0x0010
OP_OR = 0x0011
OP_XOR = 0x0012
OP_SHL = 0x0014
OP_SHR = 0x0015
OP_CMPP = 0x0021
OP_LD = 0x0030
OP_ST = 0x0031
OP_BR = 0x0040
OP_BRX = 0x0041
OP_HALT = 0x0045
OP_CPY = 0x0054
OP_LOADI = 0x0055
OP_LOADI64 = 0x0056

CONTROL_OPS = {OP_BR, OP_BRX, OP_HALT}
MEMORY_OPS = {OP_LD, OP_ST}

ALU_OPS = {
    OP_ADD, OP_SUB, OP_MUL, OP_MAD, OP_FMA,
    OP_AND, OP_OR, OP_XOR, OP_SHL, OP_SHR,
}


def _pred_enabled(instr: AECInstruction) -> bool:
    # Pred/Ctrl bit 15 is pred_en. In the in-memory IR this is carried in
    # control by the current lowering implementation.
    return bool(instr.control & 0x8000)


def _gpr_reads(instr: AECInstruction) -> Set[int]:
    reads: Set[int] = set()

    if instr.opcode in {
        OP_ADD, OP_SUB, OP_MUL,
        OP_AND, OP_OR, OP_XOR, OP_SHL, OP_SHR,
        OP_CMPP,
    }:
        reads.update((instr.src1, instr.src2))

    elif instr.opcode in {OP_MAD, OP_FMA}:
        reads.update((instr.src1, instr.src2, instr.src3))

    elif instr.opcode == OP_LD:
        reads.add(instr.src1)

    elif instr.opcode == OP_ST:
        reads.update((instr.src1, instr.src2))

    elif instr.opcode == OP_CPY:
        # Values >= 0x0100 are special-register selectors, not GPRs.
        if 0 < instr.src1 < 0x0100:
            reads.add(instr.src1)

    return {reg for reg in reads if 0 < reg < 0x0100}


def _gpr_writes(instr: AECInstruction) -> Set[int]:
    if instr.opcode in {
        OP_ADD, OP_SUB, OP_MUL, OP_MAD, OP_FMA,
        OP_AND, OP_OR, OP_XOR, OP_SHL, OP_SHR,
        OP_LD, OP_CPY, OP_LOADI,
    }:
        return {instr.dst} if 0 < instr.dst < 0x0100 else set()

    if instr.opcode == OP_LOADI64:
        writes = set()
        if 0 < instr.dst < 0x0100:
            writes.add(instr.dst)
        if 0 < instr.dst + 1 < 0x0100:
            writes.add(instr.dst + 1)
        return writes

    return set()


def _pred_reads(instr: AECInstruction) -> Set[int]:
    if instr.opcode == OP_BRX:
        return {instr.pred}
    if _pred_enabled(instr):
        return {instr.pred}
    return set()


def _pred_writes(instr: AECInstruction) -> Set[int]:
    if instr.opcode == OP_CMPP:
        return {instr.dst}
    return set()


def _build_dependencies(
    body: List[AECInstruction],
) -> Tuple[List[Set[int]], List[Set[int]]]:
    n = len(body)
    predecessors: List[Set[int]] = [set() for _ in range(n)]
    successors: List[Set[int]] = [set() for _ in range(n)]

    reads = [_gpr_reads(instr) for instr in body]
    writes = [_gpr_writes(instr) for instr in body]
    pred_reads = [_pred_reads(instr) for instr in body]
    pred_writes = [_pred_writes(instr) for instr in body]

    def add_edge(i: int, j: int) -> None:
        if i == j or j in successors[i]:
            return
        successors[i].add(j)
        predecessors[j].add(i)

    for i in range(n):
        for j in range(i + 1, n):
            # GPR RAW / WAR / WAW.
            if writes[i] & reads[j]:
                add_edge(i, j)
            if reads[i] & writes[j]:
                add_edge(i, j)
            if writes[i] & writes[j]:
                add_edge(i, j)

            # Predicate RAW / WAR / WAW.
            if pred_writes[i] & pred_reads[j]:
                add_edge(i, j)
            if pred_reads[i] & pred_writes[j]:
                add_edge(i, j)
            if pred_writes[i] & pred_writes[j]:
                add_edge(i, j)

    # Preserve lowered 64-bit register-pair construction.
    #
    # C1 lowers:
    #   mul.wide.u32 -> MUL low ; LOADI high, 0
    #   add.u64      -> ADD low ; LOADI high, 0
    #
    # Treat the adjacent pair as one logical definition. The high-half write
    # must stay after the low-half operation and before any later use of either
    # half, including a global-memory access that consumes the low address.
    for i in range(n - 1):
        low_def = body[i]
        high_def = body[i + 1]

        if (
            low_def.opcode in {OP_ADD, OP_MUL}
            and high_def.opcode == OP_LOADI
            and 0 < low_def.dst < 0x00FF
            and high_def.dst == low_def.dst + 1
        ):
            add_edge(i, i + 1)

            pair_regs = {low_def.dst, high_def.dst}
            for j in range(i + 2, n):
                if pair_regs & reads[j]:
                    add_edge(i + 1, j)

    # Preserve exact memory-operation order conservatively.
    previous_memory = None
    for i, instr in enumerate(body):
        if instr.opcode in MEMORY_OPS:
            if previous_memory is not None:
                add_edge(previous_memory, i)
            previous_memory = i

    return predecessors, successors


def _priority(instr: AECInstruction, original_index: int) -> Tuple[int, int]:
    # Lower tuple sorts first.
    if instr.opcode == OP_LD:
        return (0, original_index)
    if instr.opcode in ALU_OPS:
        return (1, original_index)
    if instr.opcode == OP_CMPP:
        return (2, original_index)
    return (3, original_index)


def _schedule_block(body: List[AECInstruction]) -> List[AECInstruction]:
    if len(body) <= 1:
        return list(body)

    predecessors, successors = _build_dependencies(body)
    remaining = [len(preds) for preds in predecessors]
    ready = [i for i, count in enumerate(remaining) if count == 0]
    scheduled_indices: List[int] = []

    # Adjacent low/high definitions produced by lowering must remain adjacent
    # after scheduling:
    #   ADD/MUL Rk, ...
    #   LOADI  Rk+1, 0
    pair_second: Dict[int, int] = {}
    for i in range(len(body) - 1):
        low_def = body[i]
        high_def = body[i + 1]
        if (
            low_def.opcode in {OP_ADD, OP_MUL}
            and high_def.opcode == OP_LOADI
            and 0 < low_def.dst < 0x00FF
            and high_def.dst == low_def.dst + 1
        ):
            pair_second[i] = i + 1

    def emit(index: int) -> None:
        if index in ready:
            ready.remove(index)
        scheduled_indices.append(index)

        for succ in sorted(successors[index]):
            remaining[succ] -= 1
            if remaining[succ] == 0 and succ not in ready:
                ready.append(succ)

    while ready:
        ready.sort(key=lambda i: _priority(body[i], i))
        current = ready.pop(0)
        emit(current)

        # Schedule the high-half initialization immediately after its paired
        # low-half definition. Because the pair is adjacent in the lowered
        # stream, the high-half should have no predecessor other than the low.
        high = pair_second.get(current)
        if high is not None:
            if remaining[high] != 0:
                # Unexpected dependency shape: fall back to original order
                # instead of risking an invalid register-pair schedule.
                return list(body)
            emit(high)

    if len(scheduled_indices) != len(body):
        # A dependency-cycle indicates a modelling bug. Preserve correctness.
        return list(body)

    return [body[i] for i in scheduled_indices]


def _block_boundaries(
    instructions: List[AECInstruction],
) -> List[Tuple[int, int]]:
    boundaries: Set[int] = {0, len(instructions)}

    for i, instr in enumerate(instructions):
        if instr.opcode in {OP_BR, OP_BRX}:
            target = int(instr.immediate)
            if 0 <= target < len(instructions):
                boundaries.add(target)
            if i + 1 <= len(instructions):
                boundaries.add(i + 1)
        elif instr.opcode == OP_HALT:
            if i + 1 <= len(instructions):
                boundaries.add(i + 1)

    points = sorted(boundaries)
    return [
        (points[i], points[i + 1])
        for i in range(len(points) - 1)
        if points[i] < points[i + 1]
    ]


def schedule_instructions(
    instructions: List[AECInstruction],
) -> List[AECInstruction]:
    """List-schedule instructions independently inside each basic block."""
    output = list(instructions)

    for start, end in _block_boundaries(instructions):
        block = instructions[start:end]
        if not block:
            continue

        terminator = None
        body = block

        if block[-1].opcode in CONTROL_OPS:
            terminator = block[-1]
            body = block[:-1]

        scheduled = _schedule_block(body)
        if terminator is not None:
            scheduled.append(terminator)

        output[start:end] = scheduled

    return output
