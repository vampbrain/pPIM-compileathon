import re
import sys
import os
import subprocess

class PIMInstruction:
    """
    Custom PIM Instruction with 24-bit format:
    - 2-bit instruction type (PROG, EXE, END)
    - 6-bit pointer value
    - 6-bit reserved/blank
    - 1-bit read flag
    - 1-bit write flag
    - 9-bit row address
    """
    # Instruction types
    PROG = 0
    EXE = 1
    END = 2
    
    # Operation descriptions for each pointer value
    OPERATION_DESCRIPTIONS = {
        1: "Configure computing resources",
        2: "Initialize accumulator to 0",
        3: "Load value from Matrix A",
        4: "Load value from Matrix B",
        5: "Multiply and accumulate values",
        6: "Store result to Matrix C",
        10: "Store value in write buffer",
        11: "Write buffer to memory"
    }
    
    def __init__(self, instr_type, pointer=0, read=0, write=0, row_address=0):
        self.instr_type = instr_type
        self.pointer = pointer & 0x3F  # 6-bit
        self.read = read & 0x1  # 1-bit
        self.write = write & 0x1  # 1-bit
        self.row_address = row_address & 0x1FF  # 9-bit

    def to_binary(self):
        """Convert instruction to 24-bit binary representation"""
        # First 2 bits: instruction type
        binary = (self.instr_type & 0x3) << 22
        # Next 6 bits: pointer value
        binary |= (self.pointer & 0x3F) << 16
        # 6 bits reserved (blank)
        # Next 1 bit: read flag
        binary |= (self.read & 0x1) << 9
        # Next 1 bit: write flag
        binary |= (self.write & 0x1) << 8
        # Last 9 bits: row address
        binary |= (self.row_address & 0x1FF)
        
        return binary
    
    def to_hex(self):
        """Convert instruction to hexadecimal representation"""
        return f"0x{self.to_binary():06X}"
    
    def get_operation_description(self):
        """Get description of the operation based on instruction type and pointer"""
        if self.instr_type == PIMInstruction.PROG:
            return "Program/Configure processing elements"
        elif self.instr_type == PIMInstruction.EXE:
            return self.OPERATION_DESCRIPTIONS.get(self.pointer, "Execute custom operation")
        elif self.instr_type == PIMInstruction.END:
            return "Terminate execution"
        else:
            return "Unknown operation"
    
    def get_memory_operation(self):
        """Get description of memory operation"""
        if self.read and self.write:
            return f"Read from and write to row {self.row_address}"
        elif self.read:
            return f"Read from row {self.row_address}"
        elif self.write:
            return f"Write to row {self.row_address}"
        else:
            return ""
    
    def __str__(self):
        instr_types = ["PROG", "EXE", "END"]
        instr_type_str = instr_types[self.instr_type] if self.instr_type < 3 else f"UNKNOWN({self.instr_type})"
        
        base_str = f"{instr_type_str}, pointer={self.pointer}, read={self.read}, write={self.write}, row_addr={self.row_address} => {self.to_hex()}"
        
        # Add operation description
        op_desc = self.get_operation_description()
        mem_op = self.get_memory_operation()
        
        if op_desc or mem_op:
            base_str += " | " + op_desc
            if mem_op:
                base_str += ", " + mem_op
                
        return base_str


class MatrixOperation:
    """Base class for matrix operations"""
    def generate_instructions(self):
        """Generate PIM instructions for this operation"""
        raise NotImplementedError("Subclasses must implement generate_instructions")


class MatrixMultiply(MatrixOperation):
    """Matrix multiplication operation for the PIM architecture"""
    def __init__(self, matrix_a_rows, matrix_a_cols, matrix_b_cols, matrix_a=None, matrix_b=None):
        self.matrix_a_rows = matrix_a_rows
        self.matrix_a_cols = matrix_a_cols
        self.matrix_b_cols = matrix_b_cols
        self.matrix_a = matrix_a  # Actual matrix values if available
        self.matrix_b = matrix_b  # Actual matrix values if available
        
        # Memory layout (row addresses):
        # - Matrix A: 0 to matrix_a_rows * matrix_a_cols - 1
        # - Matrix B: matrix_a_rows * matrix_a_cols to matrix_a_rows * matrix_a_cols + matrix_a_cols * matrix_b_cols - 1
        # - Result Matrix C: Start after Matrix B
        self.matrix_a_start = 0
        self.matrix_b_start = matrix_a_rows * matrix_a_cols
        self.matrix_c_start = self.matrix_b_start + matrix_a_cols * matrix_b_cols

    def generate_memory_initialization(self):
        """Generate instructions to initialize memory with matrix values"""
        instructions = []
        
        # Only generate initialization if we have actual matrix values
        if self.matrix_a and self.matrix_b:
            # Initialize Matrix A in memory
            for i in range(self.matrix_a_rows):
                for j in range(self.matrix_a_cols):
                    # Calculate memory address for A[i][j]
                    a_row_addr = self.matrix_a_start + i * self.matrix_a_cols + j
                    
                    # Store value into memory (simplified - in reality you'd need to convert value to proper format)
                    # For demonstration, we just use the value as-is (assuming small integers)
                    value = self.matrix_a[i][j]
                    # Store value in write buffer (EXE with pointer to store operation)
                    instructions.append(PIMInstruction(PIMInstruction.EXE, pointer=10))
                    # Write buffer to memory
                    instructions.append(PIMInstruction(PIMInstruction.EXE, pointer=11, write=1, row_address=a_row_addr))
            
            # Initialize Matrix B in memory
            for i in range(self.matrix_a_cols):
                for j in range(self.matrix_b_cols):
                    # Calculate memory address for B[i][j]
                    b_row_addr = self.matrix_b_start + i * self.matrix_b_cols + j
                    
                    # Store value into memory
                    value = self.matrix_b[i][j]
                    # Store value in write buffer
                    instructions.append(PIMInstruction(PIMInstruction.EXE, pointer=10))
                    # Write buffer to memory
                    instructions.append(PIMInstruction(PIMInstruction.EXE, pointer=11, write=1, row_address=b_row_addr))
        
        return instructions

    def generate_instructions(self):
        """Generate PIM instructions for matrix multiplication"""
        instructions = []
        
        # 1. Set up the operation (PROG instruction to set up computing resources)
        instructions.append(PIMInstruction(PIMInstruction.PROG, pointer=1))
        
        # 2. Initialize memory with matrix values if provided
        if self.matrix_a and self.matrix_b:
            instructions.extend(self.generate_memory_initialization())
        
        # 3. Matrix multiplication algorithm
        for i in range(self.matrix_a_rows):
            for j in range(self.matrix_b_cols):
                # Initialize accumulator to 0 (using dedicated register via EXE)
                instructions.append(PIMInstruction(PIMInstruction.EXE, pointer=2))
                
                for k in range(self.matrix_a_cols):
                    # Read A[i][k]
                    a_row_addr = self.matrix_a_start + i * self.matrix_a_cols + k
                    instructions.append(PIMInstruction(PIMInstruction.EXE, pointer=3, read=1, row_address=a_row_addr))
                    
                    # Read B[k][j]
                    b_row_addr = self.matrix_b_start + k * self.matrix_b_cols + j
                    instructions.append(PIMInstruction(PIMInstruction.EXE, pointer=4, read=1, row_address=b_row_addr))
                    
                    # Multiply and accumulate (using dedicated operation)
                    instructions.append(PIMInstruction(PIMInstruction.EXE, pointer=5))
                
                # Write result to C[i][j]
                c_row_addr = self.matrix_c_start + i * self.matrix_b_cols + j
                instructions.append(PIMInstruction(PIMInstruction.EXE, pointer=6, write=1, row_address=c_row_addr))
        
        # 4. End the operation
        instructions.append(PIMInstruction(PIMInstruction.END))
        
        return instructions


class CppMatrixParser:
    """Parser for C++ matrix multiplication code"""
    
    def __init__(self, cpp_file_path):
        self.cpp_file_path = cpp_file_path
        self.matrix_a_rows = None
        self.matrix_a_cols = None
        self.matrix_b_cols = None
        self.matrix_a = None
        self.matrix_b = None
    
    def parse_dimensions(self):
        """Parse matrix dimensions from C++ file"""
        try:
            with open(self.cpp_file_path, 'r') as file:
                content = file.read()
                
                # Look for matrix dimension declarations
                matrix_a_rows_match = re.search(r'MATRIX_A_ROWS\s*=\s*(\d+)', content)
                matrix_a_cols_match = re.search(r'MATRIX_A_COLS\s*=\s*(\d+)', content)
                matrix_b_cols_match = re.search(r'MATRIX_B_COLS\s*=\s*(\d+)', content)
                
                if matrix_a_rows_match and matrix_a_cols_match and matrix_b_cols_match:
                    self.matrix_a_rows = int(matrix_a_rows_match.group(1))
                    self.matrix_a_cols = int(matrix_a_cols_match.group(1))
                    self.matrix_b_cols = int(matrix_b_cols_match.group(1))
                    return True
                else:
                    print("Error: Could not find matrix dimensions in the C++ file")
                    return False
        except Exception as e:
            print(f"Error parsing C++ file: {e}")
            return False
    
    def extract_matrix_values(self):
        """
        Extract actual matrix values by instrumenting the C++ file and running it
        This is a more advanced approach than just parsing the static code
        """
        if not self.matrix_a_rows or not self.matrix_a_cols or not self.matrix_b_cols:
            print("Error: Matrix dimensions not yet parsed")
            return False
        
        try:
            # Create a modified version of the C++ file that outputs matrix values in a parsable format
            base_name = os.path.splitext(self.cpp_file_path)[0]
            instrumented_file = f"{base_name}_instrumented.cpp"
            
            with open(self.cpp_file_path, 'r') as src_file:
                content = src_file.read()
            
            # Add code to output matrices in a format we can easily parse
            output_code = """
    // Output matrices in a format for parser
    std::cout << "\\n// PARSER_OUTPUT_BEGIN" << std::endl;
    
    std::cout << "// MATRIX_A" << std::endl;
    for (int i = 0; i < MATRIX_A_ROWS; i++) {
        for (int j = 0; j < MATRIX_A_COLS; j++) {
            std::cout << A[i][j];
            if (j < MATRIX_A_COLS - 1) std::cout << ",";
        }
        std::cout << std::endl;
    }
    
    std::cout << "// MATRIX_B" << std::endl;
    for (int i = 0; i < MATRIX_A_COLS; i++) {
        for (int j = 0; j < MATRIX_B_COLS; j++) {
            std::cout << B[i][j];
            if (j < MATRIX_B_COLS - 1) std::cout << ",";
        }
        std::cout << std::endl;
    }
    
    std::cout << "// PARSER_OUTPUT_END" << std::endl;
"""
            # Insert our output code before the return statement in main()
            modified_content = re.sub(r'(\s+return\s+0;\s*\})', f'{output_code}\\1', content)
            
            with open(instrumented_file, 'w') as out_file:
                out_file.write(modified_content)
            
            # Compile and run the instrumented file
            compile_cmd = f"g++ {instrumented_file} -o {base_name}_instrumented"
            print(f"Compiling: {compile_cmd}")
            subprocess.run(compile_cmd, shell=True, check=True)
            
            run_cmd = f"./{base_name}_instrumented"
            print(f"Running: {run_cmd}")
            result = subprocess.run(run_cmd, shell=True, capture_output=True, text=True, check=True)
            
            # Parse the output to extract matrix values
            output = result.stdout
            
            # Extract the section between PARSER_OUTPUT_BEGIN and PARSER_OUTPUT_END
            parser_output_match = re.search(r'// PARSER_OUTPUT_BEGIN(.*?)// PARSER_OUTPUT_END', 
                                          output, re.DOTALL)
            
            if parser_output_match:
                parser_output = parser_output_match.group(1)
                
                # Extract Matrix A values
                matrix_a_section = re.search(r'// MATRIX_A(.*?)// MATRIX_B', parser_output, re.DOTALL)
                if matrix_a_section:
                    matrix_a_lines = matrix_a_section.group(1).strip().split('\n')
                    self.matrix_a = []
                    for line in matrix_a_lines:
                        row = [int(val) for val in line.split(',')]
                        self.matrix_a.append(row)
                
                # Extract Matrix B values
                matrix_b_section = re.search(r'// MATRIX_B(.*)', parser_output, re.DOTALL)
                if matrix_b_section:
                    matrix_b_lines = matrix_b_section.group(1).strip().split('\n')
                    self.matrix_b = []
                    for line in matrix_b_lines:
                        row = [int(val) for val in line.split(',')]
                        self.matrix_b.append(row)
                
                return True
            else:
                print("Error: Could not find parser output in the instrumented program output")
                return False
                
        except subprocess.CalledProcessError as e:
            print(f"Error running instrumented C++ file: {e}")
            print(f"Stdout: {e.stdout}")
            print(f"Stderr: {e.stderr}")
            return False
        except Exception as e:
            print(f"Error extracting matrix values: {e}")
            return False


def generate_matrix_multiplication_code(matrix_a_rows, matrix_a_cols, matrix_b_cols, matrix_a=None, matrix_b=None):
    """Generate PIM instructions for matrix multiplication with given dimensions and values"""
    matrix_mult = MatrixMultiply(matrix_a_rows, matrix_a_cols, matrix_b_cols, matrix_a, matrix_b)
    instructions = matrix_mult.generate_instructions()
    
    # Format the instructions for output
    output_lines = []
    output_lines.append(f"// Matrix Multiplication: ({matrix_a_rows}x{matrix_a_cols}) * ({matrix_a_cols}x{matrix_b_cols})")
    output_lines.append(f"// Memory layout:")
    output_lines.append(f"// - Matrix A: rows {matrix_mult.matrix_a_start}-{matrix_mult.matrix_b_start-1}")
    output_lines.append(f"// - Matrix B: rows {matrix_mult.matrix_b_start}-{matrix_mult.matrix_c_start-1}")
    output_lines.append(f"// - Result Matrix C: rows {matrix_mult.matrix_c_start}+")
    output_lines.append(f"// Total instructions: {len(instructions)}")
    
    # Add actual matrix values if available
    if matrix_a and matrix_b:
        output_lines.append(f"\n// Matrix A:")
        for row in matrix_a:
            output_lines.append(f"// {row}")
        
        output_lines.append(f"\n// Matrix B:")
        for row in matrix_b:
            output_lines.append(f"// {row}")
        
        # Add a description of the arithmetic operations
        output_lines.append(f"\n// Arithmetic Operations:")
        output_lines.append(f"// - For each element C[i][j] of the result matrix:")
        output_lines.append(f"//   1. Initialize accumulator to 0")
        output_lines.append(f"//   2. For k = 0 to {matrix_a_cols-1}:")
        output_lines.append(f"//      a. Load A[i][k]")
        output_lines.append(f"//      b. Load B[k][j]")
        output_lines.append(f"//      c. Compute A[i][k] * B[k][j] and add to accumulator")
        output_lines.append(f"//   3. Store accumulator to C[i][j]")
    
    output_lines.append("")
    
    for i, instr in enumerate(instructions):
        output_lines.append(f"{i:04d}: {instr}")
    
    return "\n".join(output_lines)


def save_to_file(content, filename):
    """Save content to file"""
    with open(filename, 'w') as f:
        f.write(content)
    print(f"Output saved to {filename}")


def main():
    """Main function to handle command line arguments and run the compiler"""
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python pim_compiler.py <cpp_file> [output_file]")
        print("Example: python pim_compiler.py matrix_multiply.cpp output.pimisa")
        # Run with default values if no arguments provided
        print("\nRunning with default values (3x3 * 3x2 matrices)")
        instructions = generate_matrix_multiplication_code(3, 3, 2)
        save_to_file(instructions, "matrix_multiplication.pimisa")
        return
    
    cpp_file = sys.argv[1]
    output_file = "matrix_multiplication.pimisa" if len(sys.argv) < 3 else sys.argv[2]
    
    print(f"Parsing C++ file: {cpp_file}")
    
    # Create parser and parse the C++ file
    parser = CppMatrixParser(cpp_file)
    
    # Parse dimensions
    if not parser.parse_dimensions():
        print("Using default matrix dimensions")
        instructions = generate_matrix_multiplication_code(3, 3, 2)
        save_to_file(instructions, output_file)
        return
    
    print(f"Detected matrix dimensions: ({parser.matrix_a_rows}x{parser.matrix_a_cols}) * ({parser.matrix_a_cols}x{parser.matrix_b_cols})")
    
    # Try to extract actual matrix values
    matrix_values_extracted = parser.extract_matrix_values()
    
    if matrix_values_extracted:
        print("Successfully extracted matrix values:")
        print("Matrix A:")
        for row in parser.matrix_a:
            print(row)
        print("Matrix B:")
        for row in parser.matrix_b:
            print(row)
        
        # Generate instructions with actual matrix values
        instructions = generate_matrix_multiplication_code(
            parser.matrix_a_rows, 
            parser.matrix_a_cols, 
            parser.matrix_b_cols,
            parser.matrix_a,
            parser.matrix_b
        )
    else:
        print("Could not extract matrix values, generating instructions with dimensions only")
        # Generate instructions with dimensions only
        instructions = generate_matrix_multiplication_code(
            parser.matrix_a_rows, 
            parser.matrix_a_cols, 
            parser.matrix_b_cols
        )
    
    # Save the generated instructions to the output file
    save_to_file(instructions, output_file)


if __name__ == "__main__":
    main()