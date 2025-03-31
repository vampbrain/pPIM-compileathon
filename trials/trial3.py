import os
import sys
import re
from pycparser import c_parser, c_ast
from llvmlite import ir, binding

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
                if a_cols != b_rows:
                    print(f"Warning: Matrix multiplication compatibility issue: A columns ({a_cols}) must match B rows ({b_rows}).")
                    # Adjust B dimensions
                    self.matrix_sizes['B'] = (a_cols, b_cols)
                    
                if a_rows != c_rows or b_cols != c_cols:
                    print(f"Warning: Result matrix C dimensions should be ({a_rows}x{b_cols}).")
                    # Adjust C dimensions
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
        if isinstance(loop_node.init, (c_ast.DeclList, c_ast.Assignment)):
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
    
    # Look for dimension declarations like "const int MATRIX_A_ROWS = 3;"
    dim_pattern = r'const\s+int\s+MATRIX_([A-Z])_([A-Z]+)\s*=\s*(\d+)'
    for match in re.finditer(dim_pattern, cpp_code):
        matrix = match.group(1)
        dim_type = match.group(2)
        value = int(match.group(3))
        
        if matrix not in matrix_sizes:
            matrix_sizes[matrix] = {}
        matrix_sizes[matrix][dim_type] = value
    
    # Convert to the expected format
    result = {}
    for matrix, dims in matrix_sizes.items():
        if 'ROWS' in dims and 'COLS' in dims:
            result[matrix] = (dims['ROWS'], dims['COLS'])
    
    # Set default values for missing matrices
    if 'A' not in result:
        result['A'] = (3, 3)
    
    # For matrix B, if not found directly, try to infer from the code
    if 'B' not in result:
        # Look for B initialization like "std::vector<std::vector<int>> B(MATRIX_A_COLS, std::vector<int>(MATRIX_B_COLS));"
        b_init_pattern = r'B\s*\(\s*MATRIX_([A-Z])_([A-Z]+)\s*,\s*std::vector\s*<\s*int\s*>\s*\(\s*MATRIX_([A-Z])_([A-Z]+)\s*\)\s*\)'
        b_match = re.search(b_init_pattern, cpp_code)
        
        if b_match and 'A' in result:
            # B is typically initialized with A's columns as its rows
            result['B'] = (result['A'][1], 3)  # Default columns if not found
    
    # Ensure B is always present
    if 'B' not in result:
        if 'A' in result:
            result['B'] = (result['A'][1], 3)  # Make B compatible with A
        else:
            result['B'] = (3, 3)  # Default size
    
    # Calculate C dimensions
    result['C'] = (result['A'][0], result['B'][1])
    
    return result, loop_structure

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
    def __init__(self, matrix_sizes, start_addr=0x100, alignment=4, rows_per_subarray=512):
        self.matrix_sizes = matrix_sizes
        self.start_addr = start_addr
        self.alignment = alignment
        self.rows_per_subarray = rows_per_subarray
        self.memory_map = self._create_memory_map()
        
    def _create_memory_map(self):
        """Create an optimized memory map for matrices"""
        memory_map = {}
        current_addr = self.start_addr
        
        # Improved memory mapping strategy: distribute matrices across subarrays
        # to enable parallel operations when possible
        matrices = sorted(['A', 'B', 'C'], 
                         key=lambda m: self.matrix_sizes.get(m, (0, 0))[0] * self.matrix_sizes.get(m, (0, 0))[1],
                         reverse=True)
        
        total_subarrays = 1  # Start with at least 1 subarray
        total_size = sum(self.matrix_sizes.get(m, (0, 0))[0] * self.matrix_sizes.get(m, (0, 0))[1] * 4 
                         for m in matrices)
        
        # Estimate number of subarrays needed
        est_subarrays = max(1, total_size // (self.rows_per_subarray * 4))
        
        for i, matrix_name in enumerate(matrices):
            if matrix_name in self.matrix_sizes:
                rows, cols = self.matrix_sizes[matrix_name]
                size = rows * cols * 4  # Assuming 4 bytes per integer
                
                # Assign to different subarrays for parallel access when possible
                target_subarray = i % max(1, est_subarrays)
                subarray_start = self.start_addr + (target_subarray * self.rows_per_subarray * 4)
                
                # Align address
                if subarray_start % self.alignment != 0:
                    subarray_start += self.alignment - (subarray_start % self.alignment)
                
                memory_map[matrix_name] = {
                    'start': subarray_start,
                    'end': subarray_start + size - 1,
                    'rows': rows,
                    'cols': cols,
                    'element_size': 4,
                    'subarray_id': target_subarray,
                    'row_offset': 0  # Will be updated later
                }
                
                # Calculate actual row offset within subarray
                memory_map[matrix_name]['row_offset'] = ((memory_map[matrix_name]['start'] - self.start_addr) // 4) % self.rows_per_subarray
        
        return memory_map
    
    def get_element_address(self, matrix, row, col):
        """Get the physical address of a matrix element"""
        if matrix not in self.memory_map:
            raise ValueError(f"Matrix {matrix} not found in memory map")
            
        info = self.memory_map[matrix]
        element_offset = (row * info['cols'] + col) * info['element_size']
        
        physical_addr = info['start'] + element_offset
        row_addr = (info['row_offset'] + (element_offset // 4)) % self.rows_per_subarray
        
        return {
            'physical_addr': physical_addr,
            'row_addr': row_addr,
            'subarray_id': info['subarray_id'] + ((info['row_offset'] + (element_offset // 4)) // self.rows_per_subarray)
        }
    
    def get_memory_map(self):
        """Return the memory map"""
        return self.memory_map

class PIMInstructionGenerator:
    def __init__(self, memory_mapper):
        self.memory_mapper = memory_mapper
        self.instructions = []
        
        # Define instruction types as per pPIM ISA spec - FIXED encoding
        # 2-bit instruction type + 6-bit pointer
        self.instruction_types = {
            'PROG': '01',  # Corrected from '00' to '01'
            'EXE': '10',   
            'END': '11'    
        }
        
        # MAC operation steps tracking
        self.current_mac_step = 0
        self.mac_operations = 0

    def generate_instructions(self, matrix_sizes):
        """Generate optimized PIM instructions for matrix multiplication"""
        a_rows, a_cols = matrix_sizes['A']
        b_rows, b_cols = matrix_sizes['B']
        c_rows, c_cols = matrix_sizes['C']
        
        # Program the LUT for matrix multiplication - assume this sets up MAC operations
        self.add_instruction('PROG', pointer=1, read=0, write=0, row_addr=0)
        
        # Determine optimal block size based on matrix dimensions and hardware constraints
        if max(a_rows, a_cols, b_rows, b_cols) > 64:
            block_size = 4
        elif max(a_rows, a_cols, b_rows, b_cols) > 16:
            block_size = 8
        else:
            block_size = 16
            
        # Make sure block size doesn't exceed matrix dimensions
        block_size = min(block_size, a_rows, a_cols, b_rows, b_cols)
        
        # Initialize instruction statistics
        self.mac_operations = 0
        
        # Initialize all elements of C to zero in one operation if possible
        self._initialize_c_matrix(c_rows, c_cols)
        
        # Loop over blocks
        for i_block in range(0, c_rows, block_size):
            i_end = min(i_block + block_size, c_rows)
            for j_block in range(0, c_cols, block_size):
                j_end = min(j_block + block_size, c_cols)
                for k_block in range(0, a_cols, block_size):
                    k_end = min(k_block + block_size, a_cols)
                    # Process the current block
                    self._process_block(i_block, i_end, j_block, j_end, k_block, k_end)
        
        # End program
        self.add_instruction('END', pointer=3, read=0, write=0, row_addr=0)
        
        return self.instructions
    
    def _initialize_c_matrix(self, c_rows, c_cols):
        """Initialize all elements of C matrix to zero"""
        memory_map = self.memory_mapper.get_memory_map()
        if 'C' in memory_map:
            c_info = memory_map['C']
            size = c_rows * c_cols
            if size <= 512:
                for i in range(c_rows):
                    for j in range(0, c_cols, 4):
                        addr_info = self.memory_mapper.get_element_address('C', i, j)
                        self.add_instruction('EXE', pointer=2, read=0, write=1, row_addr=addr_info['row_addr'])
            else:
                for i in range(0, c_rows, 4):
                    for j in range(0, c_cols, 4):
                        for ii in range(i, min(i+4, c_rows)):
                            addr_info = self.memory_mapper.get_element_address('C', ii, j)
                            self.add_instruction('EXE', pointer=2, read=0, write=1, row_addr=addr_info['row_addr'])
    
    def _process_block(self, i_start, i_end, j_start, j_end, k_start, k_end):
        """Process a block of the matrix multiplication"""
        a_elements_loaded = set()
        b_elements_loaded = set()
        
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                c_info = self.memory_mapper.get_element_address('C', i, j)
                self.add_instruction('EXE', pointer=2, read=1, write=0, row_addr=c_info['row_addr'])
                for k in range(k_start, k_end):
                    if (i, k) not in a_elements_loaded:
                        a_info = self.memory_mapper.get_element_address('A', i, k)
                        self.add_instruction('EXE', pointer=2, read=1, write=0, row_addr=a_info['row_addr'])
                        a_elements_loaded.add((i, k))
                    if (k, j) not in b_elements_loaded:
                        b_info = self.memory_mapper.get_element_address('B', k, j)
                        self.add_instruction('EXE', pointer=2, read=1, write=0, row_addr=b_info['row_addr'])
                        b_elements_loaded.add((k, j))
                    self.add_instruction('EXE', pointer=2, read=0, write=0, row_addr=0)
                    self.mac_operations += 1
                c_info = self.memory_mapper.get_element_address('C', i, j)
                self.add_instruction('EXE', pointer=2, read=0, write=1, row_addr=c_info['row_addr'])
                if len(a_elements_loaded) > 100:
                    a_elements_loaded.clear()
                if len(b_elements_loaded) > 100:
                    b_elements_loaded.clear()
    
    def add_instruction(self, instr_type, pointer, read, write, row_addr):
        """Create a pPIM instruction following the specified format"""
        op_segment = self.instruction_types[instr_type] + format(pointer, '06b')
        mem_segment = str(read) + str(write) + format(min(row_addr, 511), '09b')
        reserved = '000000'
        instruction = op_segment + reserved + mem_segment
        self.instructions.append(instruction)
        return instruction

    def format_instructions_readable(self):
        """Convert binary instructions to human-readable format"""
        readable = []
        for instr in self.instructions:
            op_type = instr[:2]
            pointer = int(instr[2:8], 2)
            read_bit = int(instr[14:15])
            write_bit = int(instr[15:16])
            row_addr = int(instr[16:], 2)
            if op_type == self.instruction_types['PROG']:
                readable.append(f"PROG, pointer={pointer}, read={read_bit}, write={write_bit}, row_addr={row_addr} => 0x{int(instr, 2):06X}")
            elif op_type == self.instruction_types['EXE']:
                if read_bit == 1 and write_bit == 0:
                    readable.append(f"EXE, pointer={pointer}, read={read_bit}, write={write_bit}, row_addr={row_addr} => 0x{int(instr, 2):06X}")
                elif read_bit == 0 and write_bit == 1:
                    readable.append(f"MEM_WRITE 0x{row_addr:X}")
                elif read_bit == 1 and write_bit == 1:
                    readable.append(f"MEM_ACCESS 0x{row_addr:X}")
                else:
                    readable.append(f"EXE 0x{pointer:02X}")
            elif op_type == self.instruction_types['END']:
                readable.append(f"END 0x{pointer:02X}")
            else:
                readable.append(f"UNKNOWN {instr}")
        return readable

    def generate_binary_instructions(self):
        """Convert instructions to binary format for hardware execution"""
        return [format(int(instr, 2), '06X') for instr in self.instructions]

    def get_instruction_stats(self):
        """Return statistics about generated instructions"""
        stats = {
            'total_instructions': len(self.instructions),
            'prog_instructions': sum(1 for instr in self.instructions if instr[:2] == self.instruction_types['PROG']),
            'exe_instructions': sum(1 for instr in self.instructions if instr[:2] == self.instruction_types['EXE']),
            'end_instructions': sum(1 for instr in self.instructions if instr[:2] == self.instruction_types['END']),
            'memory_reads': sum(1 for instr in self.instructions if instr[14:16] in ['10', '11']),
            'memory_writes': sum(1 for instr in self.instructions if instr[14:16] == '01'),
            'mac_operations': self.mac_operations
        }
        return stats

def save_output(pim_instructions, readable_instructions, memory_map, llvm_module, instruction_stats=None):
    """
    Save compilation output to files
    """
    with open(os.path.join(project_name, "binary_instructions.txt"), "w") as f:
        for instr in pim_instructions:
            f.write(instr + "\n")
    
    with open(os.path.join(project_name, "readable_instructions.txt"), "w") as f:
        for instr in readable_instructions:
            f.write(instr + "\n")
    
    with open(os.path.join(project_name, "memory_map.txt"), "w") as f:
        for matrix, info in memory_map.items():
            f.write(f"Matrix {matrix}: Start=0x{info['start']:X}, End=0x{info['end']:X}, "
                    f"Rows={info['rows']}, Cols={info['cols']}, Element Size={info['element_size']}\n")
            if 'subarray_id' in info:
                f.write(f"  Subarray ID: {info['subarray_id']}, Row Offset: {info['row_offset']}\n")
    
    with open(os.path.join(project_name, "matrix_multiply.ll"), "w") as f:
        f.write(str(llvm_module))
    
    if instruction_stats:
        with open(os.path.join(project_name, "instruction_stats.txt"), "w") as f:
            f.write("Instruction Statistics:\n")
            for stat, value in instruction_stats.items():
                f.write(f"{stat}: {value}\n")
            if 'total_instructions' in instruction_stats and instruction_stats['total_instructions'] > 0:
                compute_ratio = instruction_stats.get('exe_instructions', 0) / instruction_stats['total_instructions']
                memory_ratio = (instruction_stats.get('memory_reads', 0) + 
                                instruction_stats.get('memory_writes', 0)) / instruction_stats['total_instructions']
                f.write(f"\nEfficiency Metrics:\n")
                f.write(f"Compute to Total Instruction Ratio: {compute_ratio:.4f}\n")
                f.write(f"Memory to Total Instruction Ratio: {memory_ratio:.4f}\n")
                if 'mac_operations' in instruction_stats and instruction_stats['mac_operations'] > 0:
                    instructions_per_mac = instruction_stats['total_instructions'] / instruction_stats['mac_operations']
                    f.write(f"Instructions per MAC Operation: {instructions_per_mac:.4f}\n")

def compile_matrix_multiply(cpp_file_path):
    """
    Main compilation function that takes a C++ file path or code string and produces PIM instructions
    """
    if os.path.exists(cpp_file_path):
        with open(cpp_file_path, 'r') as f:
            cpp_code = f.read()
        print(f"Parsing C++ code from {cpp_file_path}...")
    else:
        cpp_code = cpp_file_path
        print("Parsing C++ code...")
    
    matrix_sizes, loop_structure = parse_cpp_code(cpp_code)
    
    if 'A' in matrix_sizes and 'B' in matrix_sizes:
        a_rows, a_cols = matrix_sizes['A']
        b_rows, b_cols = matrix_sizes['B']
        if a_cols != b_rows:
            print(f"Warning: Matrix dimensions incompatible for multiplication. Adjusting B dimensions.")
            matrix_sizes['B'] = (a_cols, b_cols)
        matrix_sizes['C'] = (a_rows, b_cols)
    
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
    binary_instructions = instruction_generator.generate_binary_instructions()
    instruction_stats = instruction_generator.get_instruction_stats()
    
    total_instr = len(pim_instructions)
    a_rows, a_cols = matrix_sizes['A']
    b_rows, b_cols = matrix_sizes['B']
    c_rows, c_cols = matrix_sizes['C']
    
    print(f"Generated {total_instr} instructions")
    print(f"Matrix multiplication size: ({a_rows}x{a_cols}) * ({b_rows}x{b_cols}) = ({c_rows}x{c_cols})")
    print(f"Memory operations: {instruction_stats['memory_reads']} reads, {instruction_stats['memory_writes']} writes")
    print(f"MAC operations: {instruction_stats['mac_operations']}")
    
    return pim_instructions, readable_instructions, memory_map, llvm_module, instruction_stats, binary_instructions

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python pim_compiler.py <cpp_file_path>")
        sys.exit(1)
    
    cpp_file_path = sys.argv[1]
    
    if not os.path.exists(cpp_file_path):
        print(f"Error: File {cpp_file_path} not found.")
        sys.exit(1)
    
    try:
        pim_instructions, readable_instructions, memory_map, llvm_module, instruction_stats, binary_instructions = compile_matrix_multiply(cpp_file_path)
        save_output(pim_instructions, readable_instructions, memory_map, llvm_module, instruction_stats)
        with open(os.path.join(project_name, "binary_instructions_hex.txt"), "w") as f:
            for instr in binary_instructions:
                f.write(instr + "\n")
        
        print(f"Compilation successful! Output saved to {project_name} directory")
        print(f"Generated {len(pim_instructions)} instructions")
        print("Memory map:")
        for matrix, info in memory_map.items():
            print(f" Matrix {matrix}: {info['rows']}x{info['cols']} @ 0x{info['start']:X}")
        
    except Exception as e:
        print(f"Compilation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
