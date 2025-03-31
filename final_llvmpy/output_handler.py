import os

def save_output(project_name, pim_instructions, readable_instructions, memory_map, llvm_module, instruction_stats=None):
    """
    Save compilation output to files
    """
    # Ensure project directory exists
    os.makedirs(project_name, exist_ok=True)
    
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
            if 'subarray_id' in info:
                f.write(f"  Subarray ID: {info['subarray_id']}, Row Offset: {info['row_offset']}\n")
    
    # Save LLVM IR
    with open(os.path.join(project_name, "matrix_multiply.ll"), "w") as f:
        f.write(str(llvm_module))
    
    # Save instruction statistics if provided
    if instruction_stats:
        with open(os.path.join(project_name, "instruction_stats.txt"), "w") as f:
            f.write("Instruction Statistics:\n")
            for stat, value in instruction_stats.items():
                f.write(f"{stat}: {value}\n")
            
            # Calculate efficiency metrics
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

def save_binary_instructions(project_name, binary_instructions):
    """
    Save binary instructions in hexadecimal format
    """
    # Ensure project directory exists
    os.makedirs(project_name, exist_ok=True)
    
    # Save binary instructions in hex format
    with open(os.path.join(project_name, "binary_instructions_hex.txt"), "w") as f:
        for instr in binary_instructions:
            f.write(instr + "\n")

def print_compilation_summary(instruction_stats, memory_map, matrix_sizes):
    """
    Print a summary of the compilation results
    """
    # Print statistics
    total_instr = instruction_stats['total_instructions']
    a_rows, a_cols = matrix_sizes['A']
    b_rows, b_cols = matrix_sizes['B']
    c_rows, c_cols = matrix_sizes['C']
    
    print(f"Generated {total_instr} instructions")
    print(f"Matrix multiplication size: ({a_rows}x{a_cols}) * ({b_rows}x{b_cols}) = ({c_rows}x{c_cols})")
    print(f"Memory operations: {instruction_stats['memory_reads']} reads, {instruction_stats['memory_writes']} writes")
    print(f"MAC operations: {instruction_stats['mac_operations']}")
    
    # Print memory map summary
    print("Memory map:")
    for matrix, info in memory_map.items():
        print(f" Matrix {matrix}: {info['rows']}x{info['cols']} @ 0x{info['start']:X}")