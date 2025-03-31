from llvmlite import ir, binding

# LLVM initialization
binding.initialize()
binding.initialize_native_target()
binding.initialize_native_asmprinter()

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