import re
from pycparser import c_parser, c_ast

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