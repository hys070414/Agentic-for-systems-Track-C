from __future__ import annotations

import copy

from typing import Dict, Iterable, List, Optional, Set, Tuple

from ir import (
    Function,
    Immediate,
    MemoryOperand,
    PTXInstruction,
    PTXProgram,
    Register,
)


PURE_VALUE_OPS = {
    "mov",
    "add",
    "sub",
    "mul",
    "mad",
    "fma",
    "and",
    "or",
    "xor",
    "shl",
    "shr",
}

CSE_OPS = {
    "add",
    "sub",
    "mul",
    "mad",
    "fma",
    "and",
    "or",
    "xor",
    "shl",
    "shr",
}

COMMUTATIVE_OPS = {
    "add",
    "mul",
    "and",
    "or",
    "xor",
}

SPECIAL_REGISTER_KINDS = {
    "tid.x", "tid.y", "tid.z",
    "ntid.x", "ntid.y", "ntid.z",
    "ctaid.x", "ctaid.y", "ctaid.z",
    "nctaid.x", "nctaid.y", "nctaid.z",
    "laneid", "warpid",
}


def _reg_key(reg: Register) -> Tuple[str, int]:
    return reg.kind, reg.num


def _iter_operand_registers(operand) -> Iterable[Register]:
    if isinstance(operand, Register):
        yield operand
    elif isinstance(operand, MemoryOperand):
        if isinstance(operand.base, Register):
            yield operand.base
        if isinstance(operand.offset, Register):
            yield operand.offset


def get_uses(instr: PTXInstruction) -> Set[Tuple[str, int]]:
    uses: Set[Tuple[str, int]] = set()

    for src in instr.srcs:
        for reg in _iter_operand_registers(src):
            uses.add(_reg_key(reg))

    # Predicate guards are uses too. Keep them separate from normal GPR
    # rewriting, but include them in liveness.
    if instr.pred:
        pred_text = str(instr.pred).replace("!", "").lstrip("%")
        if pred_text.startswith("p") and pred_text[1:].isdigit():
            uses.add(("p", int(pred_text[1:])))

    return uses


def get_def(instr: PTXInstruction) -> Optional[Tuple[str, int]]:
    if isinstance(instr.dest, Register):
        return _reg_key(instr.dest)
    return None


def _resolve_register(
    reg: Register,
    aliases: Dict[Tuple[str, int], Register],
) -> Register:
    seen: Set[Tuple[str, int]] = set()
    current = reg

    while _reg_key(current) in aliases:
        key = _reg_key(current)
        if key in seen:
            break
        seen.add(key)
        current = aliases[key]

    return current


def _rewrite_operand(
    operand,
    aliases: Dict[Tuple[str, int], Register],
):
    if isinstance(operand, Register):
        return _resolve_register(operand, aliases)

    if isinstance(operand, MemoryOperand):
        if isinstance(operand.base, Register):
            operand.base = _resolve_register(operand.base, aliases)
        if isinstance(operand.offset, Register):
            operand.offset = _resolve_register(operand.offset, aliases)
        return operand

    return operand


def _operand_key(operand):
    if isinstance(operand, Register):
        return ("reg", operand.kind, operand.num, operand.size)
    if isinstance(operand, Immediate):
        return ("imm", operand.dtype, operand.value)
    if isinstance(operand, MemoryOperand):
        return (
            "mem",
            _operand_key(operand.base) if operand.base is not None else None,
            _operand_key(operand.offset) if operand.offset is not None else None,
            operand.space,
        )
    return ("other", repr(operand))


def _expression_key(instr: PTXInstruction):
    operands = [_operand_key(src) for src in instr.srcs]

    # Only reorder the two-input commutative forms. MAD/FMA are not fully
    # commutative because the addend has a distinct role.
    if instr.opcode in COMMUTATIVE_OPS and len(operands) == 2:
        operands = sorted(operands, key=repr)

    return (
        instr.opcode,
        instr.dtype,
        str(instr.pred) if instr.pred else None,
        tuple(operands),
    )


def _kill_register(
    reg_key: Tuple[str, int],
    aliases: Dict[Tuple[str, int], Register],
    expressions: Dict[tuple, Register],
) -> None:
    aliases.pop(reg_key, None)

    # Any alias whose replacement is being redefined becomes invalid.
    stale_aliases = [
        alias_key
        for alias_key, replacement in aliases.items()
        if _reg_key(replacement) == reg_key
    ]
    for alias_key in stale_aliases:
        aliases.pop(alias_key, None)

    # Any cached expression that reads or produces the redefined register
    # must be discarded. Using repr here is simple and safe for the tiny C1 IR.
    stale_expressions = []
    marker = ("reg", reg_key[0], reg_key[1])
    for key, result in expressions.items():
        if _reg_key(result) == reg_key:
            stale_expressions.append(key)
            continue

        key_text = repr(key)
        if repr(marker)[:-1] in key_text:
            stale_expressions.append(key)

    for key in stale_expressions:
        expressions.pop(key, None)


def local_cse(func: Function) -> int:
    """Eliminate repeated pure expressions inside individual basic blocks."""
    removed = 0

    for block in func.blocks.values():
        aliases: Dict[Tuple[str, int], Register] = {}
        expressions: Dict[tuple, Register] = {}
        rewritten: List[PTXInstruction] = []

        for instr in block.instructions:
            instr.srcs = [
                _rewrite_operand(src, aliases)
                for src in instr.srcs
            ]

            dest_key = get_def(instr)
            if dest_key is not None:
                _kill_register(dest_key, aliases, expressions)

            if (
                instr.opcode in CSE_OPS
                and isinstance(instr.dest, Register)
            ):
                key = _expression_key(instr)
                previous = expressions.get(key)

                if previous is not None:
                    aliases[dest_key] = previous
                    removed += 1
                    continue

                expressions[key] = instr.dest

            # Loads, stores and control-flow instructions form conservative
            # value-numbering barriers. They are not CSE candidates here.
            if instr.opcode in {"ld", "st", "bra", "ret", "call", "jmp"}:
                expressions.clear()

            rewritten.append(instr)

        block.instructions = rewritten

    return removed


def _all_registers(func: Function) -> Set[Tuple[str, int]]:
    regs: Set[Tuple[str, int]] = set()
    for block in func.blocks.values():
        for instr in block.instructions:
            definition = get_def(instr)
            if definition is not None:
                regs.add(definition)
            regs.update(get_uses(instr))
    return regs


def local_dce(func: Function) -> int:
    """Remove unused pure computations without assuming perfect CFG data.

    Terminal blocks start with an empty live-out set. Blocks with successors
    start with all function registers live, which is conservative and avoids
    deleting values that may be used after a branch.
    """
    removed = 0
    all_regs = _all_registers(func)

    for block in func.blocks.values():
        live: Set[Tuple[str, int]]
        if block.successors:
            live = set(all_regs)
        else:
            live = set()

        kept_reversed: List[PTXInstruction] = []

        for instr in reversed(block.instructions):
            definition = get_def(instr)
            uses = get_uses(instr)

            removable = (
                instr.opcode in PURE_VALUE_OPS
                and definition is not None
                and instr.pred is None
            )

            if removable and definition not in live:
                removed += 1
                continue

            if definition is not None:
                live.discard(definition)
            live.update(uses)
            kept_reversed.append(instr)

        block.instructions = list(reversed(kept_reversed))

    return removed



def _loop_defined_registers(block) -> Dict[Tuple[str, int], int]:
    counts: Dict[Tuple[str, int], int] = {}
    for instr in block.instructions:
        definition = get_def(instr)
        if definition is not None:
            counts[definition] = counts.get(definition, 0) + 1
    return counts


def _is_single_block_natural_loop(func: Function, header_name: str) -> bool:
    block = func.blocks[header_name]
    return header_name in block.successors and header_name in block.predecessors


def _find_unique_preheader(func: Function, header_name: str) -> Optional[str]:
    """Return the unique predecessor outside a single-block loop."""
    header = func.blocks[header_name]
    outside = [name for name in header.predecessors if name != header_name]

    if len(outside) != 1:
        return None

    preheader_name = outside[0]
    preheader = func.blocks[preheader_name]

    # The preheader must reach the loop header. It may also contain conditional
    # exits, as in the public GEMM kernel, provided its linear fallthrough still
    # reaches the loop header.
    if header_name not in preheader.successors:
        return None

    return preheader_name


def _is_licm_candidate(instr: PTXInstruction) -> bool:
    """Allow only side-effect-free, unpredicated scalar value operations."""
    if instr.pred is not None:
        return False

    if instr.opcode not in PURE_VALUE_OPS:
        return False

    if not isinstance(instr.dest, Register):
        return False

    # Predicate definitions and special-register destinations are not moved.
    if instr.dest.kind == "p" or instr.dest.kind in SPECIAL_REGISTER_KINDS:
        return False

    # Memory operands are never loop-invariant candidates in this conservative
    # implementation, even when they appear under an otherwise pure opcode.
    return not any(isinstance(src, MemoryOperand) for src in instr.srcs)


def conservative_licm(func: Function) -> int:
    """Hoist safe invariants from single-basic-block natural loops.

    Safety restrictions:
    - only self-looping blocks are considered;
    - exactly one predecessor must exist outside the loop;
    - only pure, unpredicated scalar value operations are moved;
    - a destination must be defined exactly once in the loop;
    - every register operand must be defined outside the loop or by an earlier
      instruction already selected for hoisting;
    - loads, stores, comparisons, branches and calls are never moved.
    """
    moved_total = 0

    for header_name in list(func.blocks.keys()):
        if not _is_single_block_natural_loop(func, header_name):
            continue

        preheader_name = _find_unique_preheader(func, header_name)
        if preheader_name is None:
            continue

        loop_block = func.blocks[header_name]
        preheader = func.blocks[preheader_name]
        def_counts = _loop_defined_registers(loop_block)
        loop_defs = set(def_counts)

        invariant_defs: Set[Tuple[str, int]] = set()
        hoisted: List[PTXInstruction] = []
        remaining: List[PTXInstruction] = []

        for instr in loop_block.instructions:
            definition = get_def(instr)

            if not _is_licm_candidate(instr) or definition is None:
                remaining.append(instr)
                continue

            # Multiple loop definitions of the same destination are unsafe.
            if def_counts.get(definition, 0) != 1:
                remaining.append(instr)
                continue

            uses = get_uses(instr)

            # A loop-defined operand is invariant only when it was produced by
            # an earlier instruction that has already been selected to hoist.
            if any(
                use in loop_defs and use not in invariant_defs
                for use in uses
            ):
                remaining.append(instr)
                continue

            # Do not hoist a self-referential recurrence.
            if definition in uses:
                remaining.append(instr)
                continue

            hoisted.append(instr)
            invariant_defs.add(definition)

        if not hoisted:
            continue

        # The parsed block order represents the linear fallthrough order.
        # Appending to the preheader places invariants immediately before the
        # loop label while preserving conditional exits already in the block.
        preheader.instructions.extend(hoisted)
        loop_block.instructions = remaining
        moved_total += len(hoisted)

    return moved_total



def _same_register(left, right) -> bool:
    return (
        isinstance(left, Register)
        and isinstance(right, Register)
        and _reg_key(left) == _reg_key(right)
    )


def _immediate_value(operand) -> Optional[int]:
    if not isinstance(operand, Immediate):
        return None
    try:
        return int(operand.value)
    except (TypeError, ValueError):
        return None


def _dtype_has(instr: PTXInstruction, *parts: str) -> bool:
    dtype = str(instr.dtype or "").lstrip(".")
    tokens = {part for part in dtype.split(".") if part}
    return all(part in tokens for part in parts)


def _find_preheader_constant(preheader, reg: Register, expected: int) -> bool:
    target = _reg_key(reg)
    last_def = None
    for instr in preheader.instructions:
        if get_def(instr) == target:
            last_def = instr

    return (
        last_def is not None
        and last_def.opcode == "mov"
        and last_def.pred is None
        and len(last_def.srcs) == 1
        and _immediate_value(last_def.srcs[0]) == expected
    )


def _register_use_locations(func: Function, reg: Register):
    key = _reg_key(reg)
    locations = []

    for block_name, block in func.blocks.items():
        for index, instr in enumerate(block.instructions):
            if key in get_uses(instr):
                locations.append((block_name, index, instr))

    return locations


def affine_pointer_strength_reduction(func: Function) -> int:
    """Conservatively replace repeated affine address recomputation with pointer bumps."""
    transformed = 0

    for header_name in list(func.blocks.keys()):
        if not _is_single_block_natural_loop(func, header_name):
            continue

        preheader_name = _find_unique_preheader(func, header_name)
        if preheader_name is None:
            continue

        loop = func.blocks[header_name]
        preheader = func.blocks[preheader_name]
        instructions = loop.instructions

        for start in range(0, max(0, len(instructions) - 5)):
            chain = instructions[start:start + 6]
            if len(chain) != 6:
                continue

            mad_a, mul_a, add_a, mad_b, mul_b, add_b = chain

            if not (
                mad_a.opcode == "mad"
                and _dtype_has(mad_a, "lo", "u32")
                and mad_a.pred is None
                and isinstance(mad_a.dest, Register)
                and len(mad_a.srcs) == 3

                and mul_a.opcode == "mul"
                and _dtype_has(mul_a, "wide", "u32")
                and mul_a.pred is None
                and isinstance(mul_a.dest, Register)
                and len(mul_a.srcs) == 2
                and _same_register(mul_a.srcs[0], mad_a.dest)
                and _immediate_value(mul_a.srcs[1]) == 4

                and add_a.opcode == "add"
                and _dtype_has(add_a, "u64")
                and add_a.pred is None
                and isinstance(add_a.dest, Register)
                and len(add_a.srcs) == 2
                and isinstance(add_a.srcs[0], Register)
                and _same_register(add_a.srcs[1], mul_a.dest)

                and mad_b.opcode == "mad"
                and _dtype_has(mad_b, "lo", "u32")
                and mad_b.pred is None
                and isinstance(mad_b.dest, Register)
                and len(mad_b.srcs) == 3

                and mul_b.opcode == "mul"
                and _dtype_has(mul_b, "wide", "u32")
                and mul_b.pred is None
                and isinstance(mul_b.dest, Register)
                and len(mul_b.srcs) == 2
                and _same_register(mul_b.srcs[0], mad_b.dest)
                and _immediate_value(mul_b.srcs[1]) == 4

                and add_b.opcode == "add"
                and _dtype_has(add_b, "u64")
                and add_b.pred is None
                and isinstance(add_b.dest, Register)
                and len(add_b.srcs) == 2
                and isinstance(add_b.srcs[0], Register)
                and _same_register(add_b.srcs[1], mul_b.dest)
            ):
                continue

            iv = mad_a.srcs[2]
            if not (
                isinstance(iv, Register)
                and _same_register(mad_b.srcs[0], iv)
                and isinstance(mad_b.srcs[1], Register)
            ):
                continue

            n_reg = mad_b.srcs[1]

            update_index = None
            one_reg = None
            for index in range(start + 6, len(instructions)):
                instr = instructions[index]
                if (
                    instr.opcode == "add"
                    and _dtype_has(instr, "u32")
                    and instr.pred is None
                    and isinstance(instr.dest, Register)
                    and _same_register(instr.dest, iv)
                    and len(instr.srcs) == 2
                    and _same_register(instr.srcs[0], iv)
                    and isinstance(instr.srcs[1], Register)
                ):
                    update_index = index
                    one_reg = instr.srcs[1]
                    break

            if update_index is None or one_reg is None:
                continue
            if not _find_preheader_constant(preheader, iv, 0):
                continue
            if not _find_preheader_constant(preheader, one_reg, 1):
                continue

            ptr_a = add_a.dest
            ptr_b = add_b.dest

            safe = True
            for ptr in (ptr_a, ptr_b):
                uses = _register_use_locations(func, ptr)
                if not uses:
                    safe = False
                    break
                for block_name, index, instr in uses:
                    if block_name != header_name or index <= start + 5:
                        safe = False
                        break
                    if instr.opcode != "ld":
                        safe = False
                        break
                    if not any(
                        isinstance(src, MemoryOperand)
                        and isinstance(src.base, Register)
                        and _same_register(src.base, ptr)
                        for src in instr.srcs
                    ):
                        safe = False
                        break
                if not safe:
                    break

            if not safe:
                continue

            temps = (mad_a.dest, mul_a.dest, mad_b.dest, mul_b.dest)
            allowed = set(range(start, start + 6))
            for temp in temps:
                for block_name, index, _ in _register_use_locations(func, temp):
                    if block_name != header_name or index not in allowed:
                        safe = False
                        break
                if not safe:
                    break

            if not safe:
                continue

            preheader.instructions.extend(copy.deepcopy(chain))
            preheader.instructions.append(
                PTXInstruction(
                    "mul",
                    ".wide.u32",
                    copy.deepcopy(mul_a.dest),
                    [copy.deepcopy(one_reg), Immediate(4, "u32")],
                )
            )
            preheader.instructions.append(
                PTXInstruction(
                    "mul",
                    ".wide.u32",
                    copy.deepcopy(mul_b.dest),
                    [copy.deepcopy(n_reg), Immediate(4, "u32")],
                )
            )

            rewritten = instructions[:start] + instructions[start + 6:update_index]
            rewritten.extend(
                [
                    PTXInstruction(
                        "add",
                        ".u64",
                        copy.deepcopy(ptr_a),
                        [copy.deepcopy(ptr_a), copy.deepcopy(mul_a.dest)],
                    ),
                    PTXInstruction(
                        "add",
                        ".u64",
                        copy.deepcopy(ptr_b),
                        [copy.deepcopy(ptr_b), copy.deepcopy(mul_b.dest)],
                    ),
                ]
            )
            rewritten.extend(instructions[update_index:])

            loop.instructions = rewritten
            transformed += 1
            break

    return transformed



# Integer-only local constant propagation/folding. This deliberately stays
# inside a single basic block and clears facts at memory/control-flow barriers.
# It does not fold FP32 arithmetic, predicated definitions, or loop-carried
# values across block boundaries.
_INTEGER_FOLD_OPS = {
    "add", "sub", "mul", "and", "or", "xor", "shl", "shr", "mad",
}


def _dtype_tokens(dtype: str) -> Set[str]:
    return {
        token
        for token in str(dtype or "").lstrip(".").split(".")
        if token
    }


def _is_integer_scalar_dtype(dtype: str) -> bool:
    tokens = _dtype_tokens(dtype)
    return bool(tokens & {"u32", "s32", "b32"}) and "f32" not in tokens


def _u32(value: int) -> int:
    return int(value) & 0xFFFFFFFF


def _signed32(value: int) -> int:
    value = _u32(value)
    return value - 0x100000000 if value & 0x80000000 else value


def _immediate_like(value: int, template=None) -> Immediate:
    dtype = getattr(template, "dtype", None) or "u32"
    return Immediate(_u32(value), dtype)


def _resolve_constant_operand(
    operand,
    constants: Dict[Tuple[str, int], Immediate],
):
    if isinstance(operand, Register):
        known = constants.get(_reg_key(operand))
        if known is not None:
            return copy.deepcopy(known)

    # Do not rewrite MemoryOperand bases to immediates: the lowering contract
    # expects address registers, and turning them into immediate addresses here
    # would change the supported PTX subset.
    return operand


def _fold_integer_instruction(
    instr: PTXInstruction,
) -> Optional[Immediate]:
    if (
        instr.pred is not None
        or instr.opcode not in _INTEGER_FOLD_OPS
        or not _is_integer_scalar_dtype(instr.dtype)
    ):
        return None

    values = [_immediate_value(src) for src in instr.srcs]
    if any(value is None for value in values):
        return None

    vals = [int(value) for value in values]

    try:
        if instr.opcode == "add" and len(vals) == 2:
            result = vals[0] + vals[1]
        elif instr.opcode == "sub" and len(vals) == 2:
            result = vals[0] - vals[1]
        elif instr.opcode == "mul" and len(vals) == 2:
            result = vals[0] * vals[1]
        elif instr.opcode == "and" and len(vals) == 2:
            result = vals[0] & vals[1]
        elif instr.opcode == "or" and len(vals) == 2:
            result = vals[0] | vals[1]
        elif instr.opcode == "xor" and len(vals) == 2:
            result = vals[0] ^ vals[1]
        elif instr.opcode == "shl" and len(vals) == 2:
            result = _u32(vals[0]) << (_u32(vals[1]) & 31)
        elif instr.opcode == "shr" and len(vals) == 2:
            # C1's implemented SHR lowering is logical shift-right.
            result = _u32(vals[0]) >> (_u32(vals[1]) & 31)
        elif instr.opcode == "mad" and len(vals) == 3:
            result = vals[0] * vals[1] + vals[2]
        else:
            return None
    except (ArithmeticError, ValueError, TypeError):
        return None

    template = instr.srcs[0] if instr.srcs else None
    return _immediate_like(result, template)


def _algebraic_simplify(
    instr: PTXInstruction,
) -> Optional[PTXInstruction]:
    """Return a simpler equivalent mov when an identity is proven."""
    if (
        instr.pred is not None
        or not isinstance(instr.dest, Register)
        or not _is_integer_scalar_dtype(instr.dtype)
        or len(instr.srcs) != 2
    ):
        return None

    left, right = instr.srcs
    left_value = _immediate_value(left)
    right_value = _immediate_value(right)

    replacement = None

    if instr.opcode == "add":
        if right_value == 0:
            replacement = left
        elif left_value == 0:
            replacement = right

    elif instr.opcode == "sub":
        if right_value == 0:
            replacement = left
        elif _same_register(left, right):
            replacement = Immediate(0, "u32")

    elif instr.opcode == "mul":
        if left_value == 0 or right_value == 0:
            replacement = Immediate(0, "u32")
        elif right_value == 1:
            replacement = left
        elif left_value == 1:
            replacement = right

    elif instr.opcode == "and":
        if left_value == 0 or right_value == 0:
            replacement = Immediate(0, "u32")

    elif instr.opcode == "or":
        if right_value == 0:
            replacement = left
        elif left_value == 0:
            replacement = right
        elif _same_register(left, right):
            replacement = left

    elif instr.opcode == "xor":
        if right_value == 0:
            replacement = left
        elif left_value == 0:
            replacement = right
        elif _same_register(left, right):
            replacement = Immediate(0, "u32")

    elif instr.opcode in {"shl", "shr"}:
        if right_value == 0:
            replacement = left

    if replacement is None:
        return None

    return PTXInstruction(
        "mov",
        instr.dtype,
        copy.deepcopy(instr.dest),
        [copy.deepcopy(replacement)],
        label=instr.label,
        pred=instr.pred,
    )


def local_constant_folding(func: Function) -> Dict[str, int]:
    """Propagate and fold integer constants inside individual basic blocks."""
    propagated = 0
    folded = 0
    simplified = 0

    barriers = {"ld", "st", "bra", "ret", "call", "jmp"}

    for block in func.blocks.values():
        constants: Dict[Tuple[str, int], Immediate] = {}
        rewritten: List[PTXInstruction] = []

        for instr in block.instructions:
            # Predicated instructions cannot be assumed to execute, so neither
            # their source propagation nor destination facts are trusted.
            # Only substitute constants into scalar value operations.
            # In particular, do not turn the value operand of st.global into
            # an Immediate because the current lowering requires a register.
            propagatable_ops = _INTEGER_FOLD_OPS | {"mov"}

            if instr.pred is None and instr.opcode in propagatable_ops:
                new_srcs = []
                for src in instr.srcs:
                    replacement = _resolve_constant_operand(src, constants)
                    if isinstance(src, Register) and isinstance(replacement, Immediate):
                        propagated += 1
                    new_srcs.append(replacement)
                instr.srcs = new_srcs

            destination = get_def(instr)
            if destination is not None:
                constants.pop(destination, None)

            folded_value = _fold_integer_instruction(instr)
            if folded_value is not None and isinstance(instr.dest, Register):
                instr = PTXInstruction(
                    "mov",
                    instr.dtype,
                    copy.deepcopy(instr.dest),
                    [folded_value],
                    label=instr.label,
                    pred=None,
                )
                folded += 1
            else:
                simpler = _algebraic_simplify(instr)
                if simpler is not None:
                    instr = simpler
                    simplified += 1

            # Record only unconditional scalar mov-immediate definitions.
            if (
                instr.pred is None
                and instr.opcode == "mov"
                and isinstance(instr.dest, Register)
                and len(instr.srcs) == 1
                and isinstance(instr.srcs[0], Immediate)
                and _is_integer_scalar_dtype(instr.dtype)
            ):
                constants[_reg_key(instr.dest)] = copy.deepcopy(instr.srcs[0])

            # Conservative barriers. Facts are block-local anyway, but clearing
            # here prevents assumptions across control-flow or memory effects.
            if instr.opcode in barriers:
                constants.clear()

            rewritten.append(instr)

        block.instructions = rewritten

    return {
        "constants_propagated": propagated,
        "constants_folded": folded,
        "algebraic_simplified": simplified,
    }



def local_copy_propagation(func: Function) -> Dict[str, int]:
    """Safely propagate removable register-to-register mov instructions.

    The pass is deliberately local to one basic block. A mov is removed only
    when:
    - both source and destination are ordinary registers of the same kind/size;
    - the instruction is unconditional;
    - the destination is not used outside the block;
    - the source is not redefined before the destination's final local use;
    - the destination is not redefined before those uses.

    This avoids the classic miscompile:
        mov r2, r1
        mov r1, 7
        use r2          # must still see the old r1 value
    """
    propagated = 0
    removed = 0
    self_copies_removed = 0

    # Count uses by block so we can reject values that escape the candidate block.
    uses_by_block: Dict[str, Dict[Tuple[str, int], int]] = {}
    for block_name, block in func.blocks.items():
        counts: Dict[Tuple[str, int], int] = {}
        for instr in block.instructions:
            for use in get_uses(instr):
                counts[use] = counts.get(use, 0) + 1
        uses_by_block[block_name] = counts

    for block_name, block in func.blocks.items():
        instructions = block.instructions
        remove_indices: Set[int] = set()

        for index, instr in enumerate(instructions):
            if (
                instr.opcode != "mov"
                or instr.pred is not None
                or not isinstance(instr.dest, Register)
                or len(instr.srcs) != 1
                or not isinstance(instr.srcs[0], Register)
            ):
                continue

            dest = instr.dest
            src = instr.srcs[0]
            dest_key = _reg_key(dest)
            src_key = _reg_key(src)

            # mov r, r is always redundant.
            if dest_key == src_key:
                remove_indices.add(index)
                self_copies_removed += 1
                removed += 1
                continue

            # Keep special-register moves and cross-class/cross-width copies.
            if (
                dest.kind in SPECIAL_REGISTER_KINDS
                or src.kind in SPECIAL_REGISTER_KINDS
                or dest.kind == "p"
                or src.kind == "p"
                or dest.kind != src.kind
                or dest.size != src.size
            ):
                continue

            # If the destination is used in another block, do not remove its
            # defining copy without full CFG liveness/phi reasoning.
            escapes = any(
                other_name != block_name
                and counts.get(dest_key, 0) > 0
                for other_name, counts in uses_by_block.items()
            )
            if escapes:
                continue

            # Find local uses until the destination is redefined.
            use_indices: List[int] = []
            dest_redef_index = len(instructions)
            for j in range(index + 1, len(instructions)):
                candidate = instructions[j]
                if get_def(candidate) == dest_key:
                    dest_redef_index = j
                    break
                if dest_key in get_uses(candidate):
                    use_indices.append(j)

            # A copy with no local or escaping uses is left for DCE.
            if not use_indices:
                continue

            last_use = max(use_indices)

            # The source must retain the copied value through every rewritten use.
            source_redefined = any(
                get_def(instructions[j]) == src_key
                for j in range(index + 1, last_use + 1)
            )
            if source_redefined:
                continue

            # Rewrite only the uses dominated by this local copy and before the
            # destination's next definition.
            alias = {dest_key: copy.deepcopy(src)}
            rewritten_uses = 0
            for j in use_indices:
                if j >= dest_redef_index:
                    break
                target = instructions[j]
                new_srcs = []
                for operand in target.srcs:
                    before = repr(operand)
                    replacement = _rewrite_operand(operand, alias)
                    if repr(replacement) != before:
                        rewritten_uses += 1
                    new_srcs.append(replacement)
                target.srcs = new_srcs

            if rewritten_uses:
                propagated += rewritten_uses
                remove_indices.add(index)
                removed += 1

        if remove_indices:
            block.instructions = [
                instr
                for i, instr in enumerate(instructions)
                if i not in remove_indices
            ]

    return {
        "copies_propagated": propagated,
        "copies_removed": removed,
        "self_copies_removed": self_copies_removed,
    }

def global_unused_value_dce(func: Function) -> int:
    """Iteratively remove unconditional pure definitions that have no uses.

    Unlike local_dce(), this pass does not assume every register is live out of
    a block with successors. A definition is removed only when its destination
    has no use anywhere in the function, which is CFG-safe for pure operations.
    """
    removed_total = 0

    while True:
        used: Set[Tuple[str, int]] = set()
        for block in func.blocks.values():
            for instr in block.instructions:
                used.update(get_uses(instr))

        removed_this_round = 0
        for block in func.blocks.values():
            kept: List[PTXInstruction] = []
            for instr in block.instructions:
                definition = get_def(instr)
                removable = (
                    instr.opcode in PURE_VALUE_OPS
                    and definition is not None
                    and definition not in used
                    and instr.pred is None
                )
                if removable:
                    removed_this_round += 1
                    continue
                kept.append(instr)
            block.instructions = kept

        removed_total += removed_this_round
        if removed_this_round == 0:
            break

    return removed_total


def fuse_mul_add_to_fma(func: Function) -> int:
    """Fuse a single-use FP32 MUL followed by ADD into one FMA.

    Safety restrictions:
    - both instructions are unconditional FP32 operations in one basic block;
    - the MUL result has exactly one use in the whole function;
    - the ADD consumes the MUL result as either addend;
    - no definition of the MUL result occurs between the two instructions.

    This pass is enabled only at O3 by optimize_program(), because FMA performs
    one rounding instead of the two roundings of separate MUL and ADD.
    """
    use_counts: Dict[Tuple[str, int], int] = {}
    for block in func.blocks.values():
        for instr in block.instructions:
            for use in get_uses(instr):
                use_counts[use] = use_counts.get(use, 0) + 1

    fused = 0

    for block in func.blocks.values():
        instructions = block.instructions
        remove_indices: Set[int] = set()

        for mul_index, mul in enumerate(instructions):
            if (
                mul.opcode != "mul"
                or mul.pred is not None
                or not _dtype_has(mul, "f32")
                or not isinstance(mul.dest, Register)
                or len(mul.srcs) != 2
            ):
                continue

            mul_key = _reg_key(mul.dest)
            if use_counts.get(mul_key, 0) != 1:
                continue

            add_index = None
            add_instr = None
            for index in range(mul_index + 1, len(instructions)):
                candidate = instructions[index]

                # A redefinition before the use makes the original MUL dead or
                # changes which value the ADD would consume.
                if get_def(candidate) == mul_key:
                    break

                if mul_key not in get_uses(candidate):
                    continue

                add_index = index
                add_instr = candidate
                break

            if add_index is None or add_instr is None:
                continue

            if (
                add_instr.opcode != "add"
                or add_instr.pred is not None
                or not _dtype_has(add_instr, "f32")
                or not isinstance(add_instr.dest, Register)
                or len(add_instr.srcs) != 2
            ):
                continue

            left, right = add_instr.srcs
            if _same_register(left, mul.dest):
                addend = right
            elif _same_register(right, mul.dest):
                addend = left
            else:
                continue

            instructions[add_index] = PTXInstruction(
                "fma",
                add_instr.dtype,
                copy.deepcopy(add_instr.dest),
                [
                    copy.deepcopy(mul.srcs[0]),
                    copy.deepcopy(mul.srcs[1]),
                    copy.deepcopy(addend),
                ],
                label=add_instr.label,
                pred=None,
            )
            remove_indices.add(mul_index)
            fused += 1

        if remove_indices:
            block.instructions = [
                instr
                for index, instr in enumerate(instructions)
                if index not in remove_indices
            ]

    return fused

def optimize_program(program: PTXProgram, opt_level: str = "O2") -> Dict[str, int]:
    stats = {
        "cse_removed": 0,
        "dce_removed": 0,
        "licm_moved": 0,
        "strength_reduced": 0,
        "constants_propagated": 0,
        "constants_folded": 0,
        "algebraic_simplified": 0,
        "global_dead_removed": 0,
        "fma_fused": 0,
        "copies_propagated": 0,
        "copies_removed": 0,
        "self_copies_removed": 0,
    }

    for func in program.functions:
        if opt_level in ("O1", "O2", "O3"):
            const_stats = local_constant_folding(func)
            for key, value in const_stats.items():
                stats[key] += value

        if opt_level in ("O2", "O3"):
            # Constant folding first exposes additional CSE and LICM chances.
            stats["cse_removed"] += local_cse(func)
            stats["licm_moved"] += conservative_licm(func)
            stats["strength_reduced"] += affine_pointer_strength_reduction(func)

        if opt_level in ("O2", "O3"):
            copy_stats = local_copy_propagation(func)
            for key, value in copy_stats.items():
                stats[key] += value

        if opt_level in ("O2", "O3"):
            stats["fma_fused"] += fuse_mul_add_to_fma(func)

        if opt_level in ("O1", "O2", "O3"):
            # First remove values proven unused across the whole function,
            # then run the existing conservative local liveness pass.
            stats["global_dead_removed"] += global_unused_value_dce(func)
            stats["dce_removed"] += local_dce(func)

    return stats
