import os
import sys
from pycparser import c_parser, c_ast
from llvmlite import ir, binding
import re
# Project setup
project_name = "PIM_Compiler"
os.makedirs(project_name, exist_ok=True)

# LLVM initialization
binding.initialize()
binding.initialize_native_target()
binding.initialize_native_asmprinter()

class MatrixMultVisitor(c_ast.NodeVisitor):
    """
    AST visitor that extracts matrix dimensions from matrix multiplication code.
    Looks for dimensions in array declarations and loop bounds.
    """
    def __init__(self):
        self.matrix_sizes = {}
        self.loop_structure = []
        
    def visit_FuncDef(self, node):
        if node.decl.name == "matrixMultiply":
            params = node.decl.type.args.params
            if len(params) == 3:
                # Extract dimensions from array declarations
                for i, param in enumerate(params):
                    name = param.name
                    try:
                        if isinstance(param.type, c_ast.ArrayDecl):
                            dims = []
                            temp = param.type
                            while isinstance(temp, c_ast.ArrayDecl):
                                if isinstance(temp.dim, c_ast.Constant):
                                    dims.append(int(temp.dim.value))
                                temp = temp.type
                            if len(dims) == 2:
                                self.matrix_sizes[name] = tuple(dims)
                    except Exception as e:
                        print(f"Error extracting dimensions: {e}")

                # If dimensions not found in parameters, look in loop bounds
                if not all(matrix in self.matrix_sizes for matrix in ['A', 'B', 'C']):
                    self._extract_loop_info(node.body)
                
                # Default sizes if still not found
                if 'A' not in self.matrix_sizes:
                    self.matrix_sizes['A'] = (8, 8)
                if 'B' not in self.matrix_sizes:
                    self.matrix_sizes['B'] = (8, 8)
                if 'C' not in self.matrix_sizes:
                    self.matrix_sizes['C'] = (8, 8)
                
                # Validate matrix sizes for multiplication compatibility
                a_rows, a_cols = self.matrix_sizes['A']
                b_rows, b_cols = self.matrix_sizes['B']
                c_rows, c_cols = self.matrix_sizes['C']
                
                # Ensure dimensions are compatible for matrix multiplication
                if a_cols != b_rows or a_rows != c_rows or b_cols != c_cols:
                    print("Warning: Matrix dimensions are not compatible for multiplication.")
                    print(f"A: {self.matrix_sizes['A']}, B: {self.matrix_sizes['B']}, C: {self.matrix_sizes['C']}")
                    # Adjust dimensions to make them compatible
                    self.matrix_sizes['B'] = (a_cols, b_cols)
                    self.matrix_sizes['C'] = (a_rows, b_cols)
    
    def _extract_loop_info(self, node):
        """Extract loop information from nested loops"""
        if isinstance(node, c_ast.Compound) and node.block_items:
            for stmt in node.block_items:
                if isinstance(stmt, c_ast.For):
                    self._process_loop(stmt, 0)

    def _process_loop(self, loop_node, depth):
        """Process a single loop node to extract bounds and structure"""
        # Extract loop variable and bounds
        if isinstance(loop_node.init, c_ast.DeclList) or isinstance(loop_node.init, c_ast.Assignment):
            var_name = None
            if isinstance(loop_node.init, c_ast.DeclList) and loop_node.init.decls:
                var_name = loop_node.init.decls[0].name
            elif isinstance(loop_node.init, c_ast.Assignment):
                if isinstance(loop_node.init.lvalue, c_ast.ID):
                    var_name = loop_node.init.lvalue.name
            
            upper_bound = None
            if isinstance(loop_node.cond, c_ast.BinaryOp) and loop_node.cond.op in ['<', '<=']:
                if isinstance(loop_node.cond.right, c_ast.Constant):
                    upper_bound = int(loop_node.cond.right.value)
                elif isinstance(loop_node.cond.right, c_ast.ID):
                    pass  # Handle variable bounds later if needed
            
            if var_name and upper_bound:
                self.loop_structure.append({
                    'depth': depth,
                    'var': var_name,
                    'bound': upper_bound
                })
                
                # Use loop structure to infer matrix dimensions
                if depth == 0 and len(self.loop_structure) == 1:  # First loop
                    if 'C' not in self.matrix_sizes:
                        self.matrix_sizes['C'] = [upper_bound, None]
                    if 'A' not in self.matrix_sizes:
                        self.matrix_sizes['A'] = [upper_bound, None]
                elif depth == 1 or len(self.loop_structure) == 2:  # Second loop
                    if 'C' in self.matrix_sizes and self.matrix_sizes['C'][1] is None:
                        self.matrix_sizes['C'][1] = upper_bound
                    if 'B' not in self.matrix_sizes:
                        self.matrix_sizes['B'] = [None, upper_bound]
                elif depth == 2 or len(self.loop_structure) == 3:  # Third loop
                    if 'A' in self.matrix_sizes and self.matrix_sizes['A'][1] is None:
                        self.matrix_sizes['A'][1] = upper_bound
                    if 'B' in self.matrix_sizes and self.matrix_sizes['B'][0] is None:
                        self.matrix_sizes['B'][0] = upper_bound
            
            # Process nested loops
            if isinstance(loop_node.stmt, c_ast.Compound):
                for stmt in loop_node.stmt.block_items:
                    if isinstance(stmt, c_ast.For):
                        self._process_loop(stmt, depth + 1)
            elif isinstance(loop_node.stmt, c_ast.For):
                self._process_loop(loop_node.stmt, depth + 1)


def preprocess_cpp_file(cpp_code):
    """
    Preprocess the C++ code string to extract matrix dimensions
    """
    # Extract matrix dimensions using regex
    matrix_dims = {}
    
    # Look for dimension declarations like "const int MATRIX_A_ROWS = 3;"
    dim_pattern = r'const\s+int\s+MATRIX_([A-Z])_([A-Z]+)\s*=\s*(\d+)'
    for match in re.finditer(dim_pattern, cpp_code):
        matrix = match.group(1)
        dim_type = match.group(2)
        value = int(match.group(3))
        
        if matrix not in matrix_dims:
            matrix_dims[matrix] = {}
        matrix_dims[matrix][dim_type] = value
    
    # Convert to the expected format
    result = {}
    for matrix, dims in matrix_dims.items():
        if 'ROWS' in dims and 'COLS' in dims:
            result[matrix] = (dims['ROWS'], dims['COLS'])
    
    # If we couldn't extract dimensions, use default values
    if 'A' not in result:
        result['A'] = (3, 3)
    if 'B' not in result:
        result['B'] = (3, 3)
    if 'C' not in result:
        # Infer C dimensions from A and B
        result['C'] = (result['A'][0], result['B'][1])
    
    return result


def parse_cpp_code(cpp_code):
    """
    Parse C++ code to extract matrix dimensions and loop structure
    """
    matrix_sizes = {}
    loop_structure = []
    
    # Use regex to extract matrix dimensions
    dim_pattern = r'const\s+int\s+MATRIX_([A-Z])_([A-Z]+)\s*=\s*(\d+)'
    for match in re.finditer(dim_pattern, cpp_code):
        matrix = match.group(1)
        dim_type = match.group(2)
        value = int(match.group(3))
        
        if matrix not in matrix_sizes:
            matrix_sizes[matrix] = {}
        matrix_sizes[matrix][dim_type] = value
    
    # Convert to the expected format
    for matrix, dims in matrix_sizes.items():
        if 'ROWS' in dims and 'COLS' in dims:
            matrix_sizes[matrix] = (dims['ROWS'], dims['COLS'])
    
    # Extract loop structure
    loop_pattern = r'for\s*\((.*?);(.*?);(.*?)\)'
    for match in re.finditer(loop_pattern, cpp_code, re.DOTALL):
        init, cond, inc = match.groups()
        loop_structure.append({
            'init': init.strip(),
            'cond': cond.strip(),
            'inc': inc.strip()
        })
    
    return matrix_sizes, loop_structure


def generate_llvm_ir(matrix_sizes):
    """
    Generate LLVM IR for matrix multiplication
    """
    module = ir.Module(name="matrix_multiply")
    
    # Define types
    int_type = ir.IntType(32)
    double_type = ir.DoubleType()
    
    # Extract matrix dimensions
    a_rows, a_cols = matrix_sizes.get('A', (8, 8))
    b_rows, b_cols = matrix_sizes.get('B', (8, 8))
    c_rows, c_cols = matrix_sizes.get('C', (8, 8))
    
    # Define array types
    a_type = ir.ArrayType(ir.ArrayType(int_type, a_cols), a_rows)
    b_type = ir.ArrayType(ir.ArrayType(int_type, b_cols), b_rows)
    c_type = ir.ArrayType(ir.ArrayType(int_type, c_cols), c_rows)
    
    # Define function type and create function
    func_type = ir.FunctionType(ir.VoidType(), [
        ir.PointerType(a_type),
        ir.PointerType(b_type),
        ir.PointerType(c_type)
    ])
    
    func = ir.Function(module, func_type, name="matrixMultiply")
    func.args[0].name = "A"
    func.args[1].name = "B"
    func.args[2].name = "C"
    
    # Create basic blocks
    entry_block = func.append_basic_block(name="entry")
    builder = ir.IRBuilder(entry_block)
    
    # Create loop indices
    i = builder.alloca(int_type, name="i")
    j = builder.alloca(int_type, name="j")
    k = builder.alloca(int_type, name="k")
    
    # Initialize i=0
    builder.store(ir.Constant(int_type, 0), i)
    
    # Create loop blocks
    loop_i_cond = func.append_basic_block(name="loop_i_cond")
    loop_i_body = func.append_basic_block(name="loop_i_body")
    loop_i_inc = func.append_basic_block(name="loop_i_inc")
    loop_i_end = func.append_basic_block(name="loop_i_end")
    
    # Jump to loop condition
    builder.branch(loop_i_cond)
    
    # Loop i condition
    builder.position_at_end(loop_i_cond)
    i_val = builder.load(i)
    cond_i = builder.icmp_signed('<', i_val, ir.Constant(int_type, c_rows))
    builder.cbranch(cond_i, loop_i_body, loop_i_end)
    
    # Loop i body
    builder.position_at_end(loop_i_body)
    
    # Initialize j=0
    builder.store(ir.Constant(int_type, 0), j)
    
    # Create j loop blocks
    loop_j_cond = func.append_basic_block(name="loop_j_cond")
    loop_j_body = func.append_basic_block(name="loop_j_body")
    loop_j_inc = func.append_basic_block(name="loop_j_inc")
    loop_j_end = func.append_basic_block(name="loop_j_end")
    
    # Jump to j loop condition
    builder.branch(loop_j_cond)
    
    # Loop j condition
    builder.position_at_end(loop_j_cond)
    j_val = builder.load(j)
    cond_j = builder.icmp_signed('<', j_val, ir.Constant(int_type, c_cols))
    builder.cbranch(cond_j, loop_j_body, loop_j_end)
    
    # Loop j body
    builder.position_at_end(loop_j_body)
    
    # Initialize C[i][j] = 0
    i_val = builder.load(i)
    j_val = builder.load(j)
    indices = [i_val, j_val]
    c_elem_ptr = builder.gep(func.args[2], [ir.Constant(int_type, 0), i_val, j_val])
    builder.store(ir.Constant(int_type, 0), c_elem_ptr)
    
    # Initialize k=0
    builder.store(ir.Constant(int_type, 0), k)
    
    # Create k loop blocks
    loop_k_cond = func.append_basic_block(name="loop_k_cond")
    loop_k_body = func.append_basic_block(name="loop_k_body")
    loop_k_inc = func.append_basic_block(name="loop_k_inc")
    loop_k_end = func.append_basic_block(name="loop_k_end")
    
    # Jump to k loop condition
    builder.branch(loop_k_cond)
    
    # Loop k condition
    builder.position_at_end(loop_k_cond)
    k_val = builder.load(k)
    cond_k = builder.icmp_signed('<', k_val, ir.Constant(int_type, a_cols))
    builder.cbranch(cond_k, loop_k_body, loop_k_end)
    
    # Loop k body
    builder.position_at_end(loop_k_body)
    
    # Load values from A and B
    i_val = builder.load(i)
    j_val = builder.load(j)
    k_val = builder.load(k)
    
    # A[i][k]
    a_elem_ptr = builder.gep(func.args[0], [ir.Constant(int_type, 0), i_val, k_val])
    a_elem = builder.load(a_elem_ptr)
    
    # B[k][j]
    b_elem_ptr = builder.gep(func.args[1], [ir.Constant(int_type, 0), k_val, j_val])
    b_elem = builder.load(b_elem_ptr)
    
    # Multiply A[i][k] * B[k][j]
    prod = builder.mul(a_elem, b_elem)
    
    # C[i][j] += A[i][k] * B[k][j]
    c_elem_ptr = builder.gep(func.args[2], [ir.Constant(int_type, 0), i_val, j_val])
    c_elem = builder.load(c_elem_ptr)
    sum_val = builder.add(c_elem, prod)
    builder.store(sum_val, c_elem_ptr)
    
    # Increment k
    builder.branch(loop_k_inc)
    builder.position_at_end(loop_k_inc)
    k_val = builder.load(k)
    k_next = builder.add(k_val, ir.Constant(int_type, 1))
    builder.store(k_next, k)
    builder.branch(loop_k_cond)
    
    # End k loop
    builder.position_at_end(loop_k_end)
    
    # Increment j
    builder.branch(loop_j_inc)
    builder.position_at_end(loop_j_inc)
    j_val = builder.load(j)
    j_next = builder.add(j_val, ir.Constant(int_type, 1))
    builder.store(j_next, j)
    builder.branch(loop_j_cond)
    
    # End j loop
    builder.position_at_end(loop_j_end)
    
    # Increment i
    builder.branch(loop_i_inc)
    builder.position_at_end(loop_i_inc)
    i_val = builder.load(i)
    i_next = builder.add(i_val, ir.Constant(int_type, 1))
    builder.store(i_next, i)
    builder.branch(loop_i_cond)
    
    # End i loop
    builder.position_at_end(loop_i_end)
    
    # Return
    builder.ret_void()
    
    return module


class MemoryMapper:
    """
    Class handling memory mapping for matrices in the PIM architecture
    """
    def __init__(self, matrix_sizes, start_addr=0x100, alignment=4):
        self.matrix_sizes = matrix_sizes
        self.start_addr = start_addr
        self.alignment = alignment
        self.memory_map = self._create_memory_map()
        
    def _create_memory_map(self):
        """Create an optimized memory map for matrices"""
        memory_map = {}
        current_addr = self.start_addr
        
        # Map matrices with priority to frequently accessed ones
        for matrix_name in ['C', 'A', 'B']:  # C gets priority for better locality
            if matrix_name in self.matrix_sizes:
                rows, cols = self.matrix_sizes[matrix_name]
                size = rows * cols * 4  # Assuming 4 bytes per integer
                
                # Align address if needed
                if current_addr % self.alignment != 0:
                    current_addr += self.alignment - (current_addr % self.alignment)
                
                memory_map[matrix_name] = {
                    'start': current_addr,
                    'end': current_addr + size - 1,
                    'rows': rows,
                    'cols': cols,
                    'element_size': 4
                }
                current_addr += size
        
        return memory_map
    
    def get_element_address(self, matrix, row, col):
        """Get the physical address of a matrix element"""
        if matrix not in self.memory_map:
            raise ValueError(f"Matrix {matrix} not found in memory map")
            
        info = self.memory_map[matrix]
        element_offset = (row * info['cols'] + col) * info['element_size']
        return info['start'] + element_offset
    
    def get_memory_map(self):
        """Return the memory map"""
        return self.memory_map

def generate_simd_instructions(matrix_sizes, loop_structure):
    """
    Generate SIMD instructions for matrix multiplication
    """
    instructions = []
    
    # LUT programming instruction
    instructions.append("PROG 0x01")
    
    # Unroll loops and generate SIMD instructions
    for i in range(matrix_sizes['C'][0]):
        for j in range(matrix_sizes['C'][1]):
            instructions.append(f"MEM_READ 0x{i*matrix_sizes['A'][1] + j:03X}")
            for k in range(matrix_sizes['A'][1]):
                instructions.append("EXE 0x02")  # MAC operation
            instructions.append(f"MEM_WRITE 0x{i*matrix_sizes['C'][1] + j:03X}")
    
    instructions.append("END 0x03")
    
    return instructions


class PIMInstructionGenerator:
    def __init__(self, memory_mapper):
        self.memory_mapper = memory_mapper
        self.instructions = []
        self.opcodes = {
            'PROG': '00000001',
            'EXE': '00000010',
            'END': '00000011',
            'MEM_WRITE': '00000100',
            'MEM_READ': '00000101',
        }

    def generate_instructions(self, matrix_sizes):
        # Generate SIMD instructions
        simd_instructions = generate_simd_instructions(matrix_sizes, [])
        
        # Convert to binary format
        for instr in simd_instructions:
            opcode, operand = instr.split()
            binary_instr = self.opcodes[opcode] + format(int(operand, 16), '016b')
            self.instructions.append(binary_instr)
        
        return self.instructions

    def format_instructions_readable(self):
        return [f"{instr[:8]} {int(instr[8:], 2):04X}" for instr in self.instructions]


    
    def generate_instructions(self):
        """Generate PIM instructions for matrix multiplication"""
        # Start program
        self.instructions.append(self._format_instruction('PROG', 0))
        
        # Get matrix dimensions
        memory_map = self.memory_mapper.memory_map
        a_info = memory_map['A']
        b_info = memory_map['B']
        c_info = memory_map['C']
        
        # Generate nested loops for matrix multiplication
        for i in range(c_info['rows']):
            for j in range(c_info['cols']):
                # Get address for C[i][j]
                c_addr = self.memory_mapper.get_element_address('C', i, j)
                
                # Initialize C[i][j] = 0
                self.instructions.append(self._format_instruction('MEM_WRITE', c_addr, 0))
                
                for k in range(a_info['cols']):
                    # Get addresses for A[i][k] and B[k][j]
                    a_addr = self.memory_mapper.get_element_address('A', i, k)
                    b_addr = self.memory_mapper.get_element_address('B', k, j)
                    
                    # Read A[i][k]
                    self.instructions.append(self._format_instruction('MEM_READ', a_addr))
                    
                    # Read B[k][j]
                    self.instructions.append(self._format_instruction('MEM_READ', b_addr))
                    
                    # Execute multiplication and accumulation
                    self.instructions.append(self._format_instruction('EXE', 0))
                    
                    # Write result back to C[i][j]
                    self.instructions.append(self._format_instruction('MEM_ACCESS', c_addr))
        
        # End program
        self.instructions.append(self._format_instruction('END', 0))
        
        return self.instructions
    
    def _format_instruction(self, opcode, address, value=None):
        """Format a PIM instruction"""
        if opcode in ['PROG', 'EXE', 'END']:
            return f"{self.opcodes[opcode]}000000000000"
        else:
            addr_binary = format(address, '010b')
            if len(addr_binary) > 10:
                addr_binary = addr_binary[-10:]  # Truncate to 10 bits if needed
            return f"{self.opcodes[opcode]}{addr_binary}"
    
    def format_instructions_readable(self):
        """Convert binary instructions to human-readable format"""
        readable = []
        for instr in self.instructions:
            if instr.startswith(self.opcodes['PROG']):
                readable.append("PROG 0x01")
            elif instr.startswith(self.opcodes['EXE']):
                readable.append("EXE 0x02")
            elif instr.startswith(self.opcodes['END']):
                readable.append("END 0x03")
            elif instr.startswith(self.opcodes['MEM_WRITE']):
                addr = int(instr[7:], 2)
                readable.append(f"MEM_WRITE 0x{addr:X}")
            elif instr.startswith(self.opcodes['MEM_READ']):
                addr = int(instr[7:], 2)
                readable.append(f"MEM_READ 0x{addr:X}")
            elif instr.startswith(self.opcodes['MEM_ACCESS']):
                addr = int(instr[7:], 2)
                readable.append(f"MEM_ACCESS 0x{addr:X}")
            else:
                readable.append(f"UNKNOWN {instr}")
        return readable


def compile_matrix_multiply(cpp_file_path):
    with open(cpp_file_path, 'r') as f:
        cpp_code = f.read()
    
    print(f"Parsing C++ code from {cpp_file_path}...")
    matrix_sizes, loop_structure = parse_cpp_code(cpp_code)
    print(f"Detected matrix dimensions: A{matrix_sizes['A']}, B{matrix_sizes['B']}, C{matrix_sizes['C']}")
    
    print("Generating LLVM IR...")
    llvm_module = generate_llvm_ir(matrix_sizes)
    
    print("Creating memory map...")
    memory_mapper = MemoryMapper(matrix_sizes)
    memory_map = memory_mapper.get_memory_map()
    
    print("Generating PIM instructions...")
    instruction_generator = PIMInstructionGenerator(memory_mapper)
    pim_instructions = instruction_generator.generate_instructions(matrix_sizes)
    readable_instructions = instruction_generator.format_instructions_readable()
    
    return pim_instructions, readable_instructions, memory_map, llvm_module



def save_output(pim_instructions, readable_instructions, memory_map, llvm_module):
    """
    Save compilation output to files
    """
    # Save binary instructions
    with open(os.path.join(project_name, "binary_instructions.txt"), "w") as f:
        for instr in pim_instructions:
            f.write(instr + "\n")

    # Save readable instructions
    with open(os.path.join(project_name, "readable_instructions.txt"), "w") as f:
        for instr in readable_instructions:
            f.write(instr + "\n")

    # Save memory map
    with open(os.path.join(project_name, "memory_map.txt"), "w") as f:
        for matrix, info in memory_map.items():
            f.write(f"Matrix {matrix}: Start=0x{info['start']:X}, End=0x{info['end']:X}, "
                   f"Rows={info['rows']}, Cols={info['cols']}, Element Size={info['element_size']}\n")
    
    # Save LLVM IR
    with open(os.path.join(project_name, "matrix_multiply.ll"), "w") as f:
        f.write(str(llvm_module))


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python pim_compiler.py <cpp_file>")
        sys.exit(1)
    
    cpp_file_path = sys.argv[1]
    
    # Check if file exists
    if not os.path.exists(cpp_file_path):
        print(f"Error: File {cpp_file_path} not found.")
        sys.exit(1)
    
    # Compile the C++ code
    try:
        pim_instructions, readable_instructions, memory_map, llvm_module = compile_matrix_multiply(cpp_file_path)
        
        # Save the output
        save_output(pim_instructions, readable_instructions, memory_map, llvm_module)
        
        print(f"Compilation successful! Output saved to {project_name} directory")
        print(f"Generated {len(pim_instructions)} instructions")
        print("Memory map:")
        for matrix, info in memory_map.items():
            print(f"  Matrix {matrix}: {info['rows']}x{info['cols']} @ 0x{info['start']:X}")
        
    except Exception as e:
        print(f"Compilation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()