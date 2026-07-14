from typing import List, Dict, Optional
from ptx_lexer import TokenType, Token
from ir import Register, Immediate, MemoryOperand, PTXInstruction, BasicBlock, Function, PTXProgram, KernelParam


class PTXParser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Optional[Token]:
        while self.pos < len(self.tokens) and self.tokens[self.pos].type in (TokenType.NEWLINE, TokenType.LT, TokenType.GT):
            self.pos += 1
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def consume(self) -> Token:
        while self.pos < len(self.tokens) and self.tokens[self.pos].type in (TokenType.NEWLINE, TokenType.LT, TokenType.GT):
            self.pos += 1
        token = self.tokens[self.pos]
        self.pos += 1
        return token

    def expect(self, token_type: TokenType, value: str = None) -> Token:
        token = self.consume()
        if token.type != token_type:
            raise ValueError(f"Expected {token_type}, got {token.type} at line {token.line}")
        if value and token.value != value:
            raise ValueError(f"Expected '{value}', got '{token.value}' at line {token.line}")
        return token

    def parse_immediate(self, token: Token) -> Immediate:
        value_str = token.value
        if value_str.startswith('0f') or value_str.startswith('0F'):
            return Immediate(int(value_str, 16), 'f32')
        elif value_str.startswith('0x') or value_str.startswith('0X'):
            return Immediate(int(value_str, 16), 'u32')
        else:
            try:
                return Immediate(int(value_str), 'u32')
            except ValueError:
                return Immediate(int(float(value_str) * 1000000), 'f32')

    def parse_register(self, token: Token) -> Register:
        reg_str = token.value[1:]

        # Preserve the x/y/z component of PTX special registers.
        # Examples:
        #   %tid.x   -> Register("tid.x", 0, 4)
        #   %ctaid.y -> Register("ctaid.y", 0, 4)
        # Without this special case, ".y" and ".z" are discarded and all
        # dimensions later lower incorrectly as the x dimension.
        special_registers = {
            "tid.x", "tid.y", "tid.z",
            "ntid.x", "ntid.y", "ntid.z",
            "ctaid.x", "ctaid.y", "ctaid.z",
            "nctaid.x", "nctaid.y", "nctaid.z",
            "laneid", "warpid",
        }
        if reg_str in special_registers:
            return Register(reg_str, 0, 4)

        kind_map = [
            ('ctaid', ('ctaid', 4)), ('nctaid', ('nctaid', 4)), ('warpid', ('warpid', 4)),
            ('laneid', ('laneid', 4)), ('ntid', ('ntid', 4)), ('tid', ('tid', 4)),
            ('rd', ('rd', 8)), ('f', ('f', 4)), ('h', ('h', 2)), ('r', ('r', 4)),
            ('p', ('p', 1)), ('s', ('s', 4)), ('u', ('u', 4)), ('b', ('b', 1)),
            ('c', ('c', 4)), ('v', ('v', 4))
        ]
        for kind, (short_kind, size) in kind_map:
            if reg_str.startswith(kind):
                rest = reg_str[len(kind):]
                if rest and rest[0].isdigit():
                    num = int(rest)
                else:
                    num = 0
                return Register(short_kind, num, size)
        return Register('r', int(reg_str), 4)

    def parse_memory_operand(self) -> MemoryOperand:
        self.expect(TokenType.LBRACKET)
        token = self.consume()
        base = None
        offset = None
        space = 'gmem'

        known_spaces = {'gmem', 'smem', 'pmem', 'const', '.param'}
        if token.type == TokenType.REGISTER:
            base = self.parse_register(token)
        elif token.type == TokenType.IDENTIFIER:
            if token.value in known_spaces:
                space = token.value
                token = self.consume()
                if token.type == TokenType.REGISTER:
                    base = self.parse_register(token)
                elif token.type == TokenType.IDENTIFIER:
                    base = token.value
            else:
                base = token.value

        if self.peek() and self.peek().type == TokenType.COMMA:
            self.consume()
            token = self.consume()
            if token.type == TokenType.REGISTER:
                offset = self.parse_register(token)
            else:
                offset = self.parse_immediate(token)

        self.expect(TokenType.RBRACKET)
        return MemoryOperand(base, offset, space)

    def parse_instruction(self) -> PTXInstruction:
        pred = None
        if self.peek() and self.peek().type == TokenType.AT:
            self.consume()

            negated = False
            if self.peek() and self.peek().type == TokenType.NOT:
                self.consume()
                negated = True

            token = self.expect(TokenType.REGISTER)
            pred = ("!" if negated else "") + token.value

        opcode_token = self.consume()
        opcode_parts = opcode_token.value.split('.')
        opcode = opcode_parts[0]
        dtype = opcode_parts[1] if len(opcode_parts) > 1 else ''

        while self.peek() and self.peek().type == TokenType.DOT_DIRECTIVE:
            dtype += '.' + self.consume().value[1:]

        dest = None
        srcs = []

        if opcode in ('ld', 'st', 'mov', 'add', 'sub', 'mul', 'mad', 'div', 'fma',
                      'cvt', 'setp', 'bra', 'ret', 'call', 'jmp', 'and', 'or', 'xor',
                      'not', 'shl', 'shr', 'abs', 'neg', 'min', 'max'):
            if opcode == 'st':
                token = self.consume()
                if token.type == TokenType.LBRACKET:
                    self.pos -= 1
                    mem = self.parse_memory_operand()
                    srcs.append(mem)
                else:
                    srcs.append(self.parse_register(token))
                    self.expect(TokenType.COMMA)
                    mem = self.parse_memory_operand()
                    srcs.append(mem)
                self.expect(TokenType.COMMA)
                token = self.consume()
                if token.type == TokenType.REGISTER:
                    srcs.append(self.parse_register(token))
                else:
                    srcs.append(self.parse_immediate(token))
            elif opcode == 'ld':
                token = self.consume()
                if token.type == TokenType.REGISTER:
                    dest = self.parse_register(token)
                self.expect(TokenType.COMMA)
                mem = self.parse_memory_operand()
                srcs.append(mem)
            elif opcode in ('bra', 'ret', 'call', 'jmp'):
                if self.peek() and self.peek().type != TokenType.SEMICOLON:
                    token = self.consume()
                    srcs.append(token.value)
            else:
                token = self.consume()
                if token.type == TokenType.REGISTER:
                    dest = self.parse_register(token)
                while self.peek() and self.peek().type == TokenType.COMMA:
                    self.consume()
                    token = self.consume()
                    if token.type == TokenType.REGISTER:
                        srcs.append(self.parse_register(token))
                    elif token.type == TokenType.LBRACKET:
                        self.pos -= 1
                        srcs.append(self.parse_memory_operand())
                    else:
                        srcs.append(self.parse_immediate(token))

        if self.peek() and self.peek().type == TokenType.SEMICOLON:
            self.consume()
        return PTXInstruction(opcode, dtype, dest, srcs, pred=pred)

    @staticmethod
    def rebuild_cfg(blocks: Dict[str, BasicBlock]) -> None:
        """Rebuild successors and predecessors after all labels are known.

        Rules:
        - every conditional branch contributes its explicit target;
        - a final unconditional branch suppresses fallthrough;
        - a final ret suppresses fallthrough;
        - otherwise the block falls through to the next block;
        - predecessor lists are derived from the completed successor lists.
        """
        ordered_names = list(blocks.keys())

        for block in blocks.values():
            block.successors = []
            block.predecessors = []

        def add_unique(items: List[str], value: str) -> None:
            if value not in items:
                items.append(value)

        for index, name in enumerate(ordered_names):
            block = blocks[name]
            has_unconditional_terminator = False
            has_return_terminator = False

            for instr in block.instructions:
                if instr.opcode == "bra":
                    target = instr.srcs[0] if instr.srcs else None
                    if target is not None:
                        add_unique(block.successors, str(target))

                    if not instr.pred:
                        has_unconditional_terminator = True

                elif instr.opcode == "ret":
                    has_return_terminator = True

            if (
                not has_unconditional_terminator
                and not has_return_terminator
                and index + 1 < len(ordered_names)
            ):
                add_unique(block.successors, ordered_names[index + 1])

        for name in ordered_names:
            block = blocks[name]
            for successor in block.successors:
                if successor not in blocks:
                    raise ValueError(
                        f"Basic block {name!r} branches to unknown label "
                        f"{successor!r}"
                    )
                add_unique(blocks[successor].predecessors, name)

    def parse_function(self) -> Function:
        self.expect(TokenType.DOT_DIRECTIVE, '.visible')
        self.expect(TokenType.DOT_DIRECTIVE, '.entry')
        func_name = self.consume().value
        self.expect(TokenType.LPAREN)

        params: List[KernelParam] = []
        while self.peek() and self.peek().type != TokenType.RPAREN:
            if (
                self.peek().type == TokenType.DOT_DIRECTIVE
                and self.peek().value == ".param"
            ):
                self.consume()  # .param
                dtype_token = self.expect(TokenType.DOT_DIRECTIVE)
                name_token = self.consume()

                dtype = dtype_token.value.lstrip(".")
                params.append(KernelParam(name_token.value, dtype))

                if self.peek() and self.peek().type == TokenType.COMMA:
                    self.consume()
            else:
                self.consume()

        self.expect(TokenType.RPAREN)
        self.expect(TokenType.LBRACE)

        registers = {}
        blocks: Dict[str, BasicBlock] = {}
        current_block = BasicBlock('entry')
        blocks['entry'] = current_block
        entry_block = 'entry'

        while self.peek() and self.peek().type != TokenType.RBRACE:
            token = self.peek()
            if token.type == TokenType.DOT_DIRECTIVE:
                directive = self.consume().value
                if directive == '.param':
                    dtype_token = self.expect(TokenType.DOT_DIRECTIVE)
                    name_token = self.consume()
                    params.append(
                        KernelParam(name_token.value, dtype_token.value.lstrip("."))
                    )
                    if self.peek() and self.peek().type == TokenType.COMMA:
                        self.consume()
                    elif self.peek() and self.peek().type == TokenType.SEMICOLON:
                        self.consume()
                elif directive == '.reg':
                    dtype_token = self.consume()
                    reg_token = self.consume()
                    reg_str = reg_token.value[1:]
                    bracket_idx = reg_str.find('<')
                    if bracket_idx != -1:
                        kind = reg_str[:bracket_idx]
                        count = int(reg_str[bracket_idx+1:-1])
                    else:
                        kind = reg_str
                        count = 1

                    size_map = {'.f32': 4, '.f64': 8, '.f16': 2, '.bf16': 2,
                                '.u32': 4, '.s32': 4, '.u64': 8, '.s64': 8,
                                '.pred': 1, '.b32': 4, '.b64': 8}
                    size = size_map.get(dtype_token.value, 4)

                    if kind not in registers:
                        registers[kind] = []
                    for i in range(count):
                        registers[kind].append(Register(kind, i + 1, size))
                    if self.peek() and self.peek().type == TokenType.SEMICOLON:
                        self.consume()
                elif directive == '.shared':
                    self.consume()
                    if self.peek() and self.peek().type == TokenType.SEMICOLON:
                        self.consume()
                elif directive == '.const':
                    self.consume()
                    if self.peek() and self.peek().type == TokenType.SEMICOLON:
                        self.consume()
            elif token.type == TokenType.LABEL:
                label_name = self.consume().value[:-1]
                new_block = BasicBlock(label_name)
                blocks[label_name] = new_block
                current_block = new_block
            elif token.type in (TokenType.AT, TokenType.IDENTIFIER):
                instr = self.parse_instruction()
                current_block.instructions.append(instr)

                # CFG edges are rebuilt after the complete function has been
                # parsed. Branch targets may refer to labels that have not
                # appeared yet, so constructing predecessors here is unsafe.
            else:
                self.consume()

        self.expect(TokenType.RBRACE)

        self.rebuild_cfg(blocks)
        return Function(func_name, params, registers, blocks, entry_block)

    def parse(self) -> PTXProgram:
        version = "1.0"
        target = "aec_sm_10"
        address_size = 64
        functions = []

        while self.peek():
            token = self.peek()
            if token.type == TokenType.DOT_DIRECTIVE:
                directive = self.consume().value
                if directive == '.version':
                    version = self.consume().value
                    if self.peek() and self.peek().type == TokenType.SEMICOLON:
                        self.consume()
                elif directive == '.target':
                    target = self.consume().value
                    if self.peek() and self.peek().type == TokenType.SEMICOLON:
                        self.consume()
                elif directive == '.address_size':
                    address_size = int(self.consume().value)
                    if self.peek() and self.peek().type == TokenType.SEMICOLON:
                        self.consume()
                elif directive == '.visible':
                    self.pos -= 1
                    functions.append(self.parse_function())
            else:
                self.consume()

        return PTXProgram(version, target, address_size, functions)


def parse_ptx(source: str) -> PTXProgram:
    from ptx_lexer import PTXLexer
    lexer = PTXLexer()
    tokens = lexer.tokenize(source)
    parser = PTXParser(tokens)
    return parser.parse()