#!/usr/bin/env python3
import os
import sys
import argparse
from matrix_parser import parse_cpp_code
from llvm_generator import generate_llvm_ir
from memory_mapper import MemoryMapper
from instruction_generator import PIMInstructionGenerator
from output_handler import save_output

def compile_matrix_multiply(cpp_file_path):
    """
    Main compilation function that takes a C++ file path or code string and produces PIM instructions
    """
    # Check if cpp_file_path is a file path or a code string
    if os.path.exists(cpp_file_path):
        with open(cpp_file_path, 'r') as f:
            cpp_code = f.read()
        print(f"Parsing C++ code from {cpp_file_path}...")
    else:
        # Assume it's already the code content
        cpp_code = cpp_file_path
        print("Parsing C++ code...")
    
    matrix_sizes, loop_structure = parse_cpp_code(cpp_code)
    
    # Ensure C matrix dimensions are properly calculated based on A and B
    if 'A' in matrix_sizes and 'B' in matrix_sizes:
        a_rows, a_cols = matrix_sizes['A']
        b_rows, b_cols = matrix_sizes['B']
        
        # Ensure compatibility for multiplication
        if a_cols != b_rows:
            print(f"Warning: Matrix dimensions incompatible for multiplication. Adjusting B dimensions.")
            matrix_sizes['B'] = (a_cols, b_cols)
        
        # Set C dimensions based on A and B
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
    
    # Print some statistics
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
    parser = argparse.ArgumentParser(description='PIM Compiler for Matrix Multiplication')
    parser.add_argument('cpp_file', help='Path to C++ source file')
    parser.add_argument('--output-dir', default='PIM_Compiler', help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if file exists
    if not os.path.exists(args.cpp_file):
        print(f"Error: File {args.cpp_file} not found.")
        sys.exit(1)
    
    # Compile the C++ code
    try:
        pim_instructions, readable_instructions, memory_map, llvm_module, instruction_stats, binary_instructions = compile_matrix_multiply(args.cpp_file)
        
        # Save the output
        save_output(args.output_dir, pim_instructions, readable_instructions, memory_map, llvm_module, instruction_stats)
        
        # Also save binary instructions separately
        with open(os.path.join(args.output_dir, "binary_instructions_hex.txt"), "w") as f:
            for instr in binary_instructions:
                f.write(instr + "\n")
        
        print(f"Compilation successful! Output saved to {args.output_dir} directory")
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