class PIMInstructionGenerator:
    def __init__(self, memory_mapper):
        self.memory_mapper = memory_mapper
        self.instructions = []
        
        # Define opcode mapping as per pPIM ISA:
        # PROG = 1, EXE = 2, END = 3.
        # These values will be placed in the lower 2 bits of the 8-bit op segment.
        self.instruction_types = {
            'PROG': 1,  
            'EXE': 2,   
            'END': 3    
        }
        
        # MAC operation steps tracking
        self.current_mac_step = 0
        self.mac_operations = 0

    def generate_instructions(self, matrix_sizes):
        """Generate optimized PIM instructions for matrix multiplication"""
        a_rows, a_cols = matrix_sizes['A']
        b_rows, b_cols = matrix_sizes['B']
        c_rows, c_cols = matrix_sizes['C']
        
        # Use fixed pointer for PROG and END instructions.
        # For EXE instructions, we maintain an exe_pointer that increments.
        exe_pointer = 0
        
        # PROG instruction (using fixed pointer 0)
        self.add_instruction('PROG', pointer=0, read=0, write=0, row_addr=0)
        
        # Determine optimal block size.
        if max(a_rows, a_cols, b_rows, b_cols) > 64:
            block_size = 4
        elif max(a_rows, a_cols, b_rows, b_cols) > 16:
            block_size = 8
        else:
            block_size = 16
        block_size = min(block_size, a_rows, a_cols, b_rows, b_cols)
        
        self.mac_operations = 0
        
        # Initialize C matrix to zero; update exe_pointer accordingly.
        exe_pointer = self._initialize_c_matrix(c_rows, c_cols, exe_pointer)
        
        # Loop over blocks and generate EXE instructions.
        for i_block in range(0, c_rows, block_size):
            i_end = min(i_block + block_size, c_rows)
            for j_block in range(0, c_cols, block_size):
                j_end = min(j_block + block_size, c_cols)
                for k_block in range(0, a_cols, block_size):
                    k_end = min(k_block + block_size, a_cols)
                    exe_pointer = self._process_block(i_block, i_end, j_block, j_end, k_block, k_end, exe_pointer)
        
        # END instruction (using fixed pointer 0)
        self.add_instruction('END', pointer=0, read=0, write=0, row_addr=0)
        
        return self.instructions
    
    def _initialize_c_matrix(self, c_rows, c_cols, exe_pointer):
        """Initialize all elements of C matrix to zero using EXE instructions.
           Returns the updated exe_pointer.
        """
        memory_map = self.memory_mapper.get_memory_map()
        if 'C' in memory_map:
            c_info = memory_map['C']
            size = c_rows * c_cols
            
            if size <= 512:
                for i in range(c_rows):
                    for j in range(0, c_cols, 4):
                        addr_info = self.memory_mapper.get_element_address('C', i, j)
                        self.add_instruction('EXE', pointer=exe_pointer, read=0, write=1, row_addr=addr_info['row_addr'])
                        exe_pointer += 1
            else:
                for i in range(0, c_rows, 4):
                    for j in range(0, c_cols, 4):
                        for ii in range(i, min(i+4, c_rows)):
                            addr_info = self.memory_mapper.get_element_address('C', ii, j)
                            self.add_instruction('EXE', pointer=exe_pointer, read=0, write=1, row_addr=addr_info['row_addr'])
                            exe_pointer += 1
        return exe_pointer
    
    def _process_block(self, i_start, i_end, j_start, j_end, k_start, k_end, exe_pointer):
        """Process a block of the matrix multiplication.
           Returns the updated exe_pointer.
        """
        a_elements_loaded = set()
        b_elements_loaded = set()
        
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                # Load C[i][j] once for this block calculation.
                c_info = self.memory_mapper.get_element_address('C', i, j)
                self.add_instruction('EXE', pointer=exe_pointer, read=1, write=0, row_addr=c_info['row_addr'])
                exe_pointer += 1
                
                for k in range(k_start, k_end):
                    if (i, k) not in a_elements_loaded:
                        a_info = self.memory_mapper.get_element_address('A', i, k)
                        self.add_instruction('EXE', pointer=exe_pointer, read=1, write=0, row_addr=a_info['row_addr'])
                        exe_pointer += 1
                        a_elements_loaded.add((i, k))
                    
                    if (k, j) not in b_elements_loaded:
                        b_info = self.memory_mapper.get_element_address('B', k, j)
                        self.add_instruction('EXE', pointer=exe_pointer, read=1, write=0, row_addr=b_info['row_addr'])
                        exe_pointer += 1
                        b_elements_loaded.add((k, j))
                    
                    # Execute MAC operation.
                    self.add_instruction('EXE', pointer=exe_pointer, read=0, write=0, row_addr=0)
                    exe_pointer += 1
                    self.mac_operations += 1
                
                # Write result to C[i][j].
                c_info = self.memory_mapper.get_element_address('C', i, j)
                self.add_instruction('EXE', pointer=exe_pointer, read=0, write=1, row_addr=c_info['row_addr'])
                exe_pointer += 1
                
                if len(a_elements_loaded) > 100:
                    a_elements_loaded.clear()
                if len(b_elements_loaded) > 100:
                    b_elements_loaded.clear()
        return exe_pointer

    def add_instruction(self, instr_type, pointer, read, write, row_addr):
        """
        Create a 24-bit instruction.
        Format:
          - 8-bit operation segment: 6-bit pointer (upper bits) and 2-bit opcode (lower bits).
            That is: op_segment = (pointer << 2) | opcode.
          - 6 reserved bits (all zeros).
          - 10-bit memory access segment: 1-bit read, 1-bit write, 8-bit row address.
            (Row address is clamped to 255.)
        """
        # Ensure pointer is within 6 bits.
        pointer_val = pointer % 64
        opcode = self.instruction_types[instr_type]  # PROG=1, EXE=2, END=3.
        op_segment_val = (pointer_val << 2) | opcode
        op_segment = format(op_segment_val, '08b')
        
        reserved = '000000'
        mem_segment = str(read) + str(write) + format(min(row_addr, 255), '08b')
        
        instruction = op_segment + reserved + mem_segment
        self.instructions.append(instruction)
        return instruction

    def format_instructions_readable(self):
        """Convert instructions to a human-readable format."""
        readable = []
        for instr in self.instructions:
            # The first 8 bits form the op segment.
            op_segment = instr[:8]
            # The opcode is in the lower 2 bits of the op segment.
            opcode_extracted = int(op_segment[-2:], 2)
            pointer_extracted = int(op_segment[:-2], 2)
            
            read_bit = int(instr[14:15])
            write_bit = int(instr[15:16])
            row_addr = int(instr[16:], 2)
            
            # Determine instruction type based on opcode.
            if opcode_extracted == self.instruction_types['PROG']:
                instr_type = "PROG"
            elif opcode_extracted == self.instruction_types['EXE']:
                instr_type = "EXE"
            elif opcode_extracted == self.instruction_types['END']:
                instr_type = "END"
            else:
                instr_type = "UNKNOWN"
            
            if instr_type == "MEM_WRITE":  # Not used; we print MEM_WRITE for specific read/write patterns.
                readable.append(f"MEM_WRITE 0x{row_addr:X}")
            else:
                readable.append(f"{instr_type}, pointer={pointer_extracted}, read={read_bit}, write={write_bit}, row_addr={row_addr} => 0x{int(instr, 2):06X}")
        return readable

    def generate_binary_instructions(self):
        """Return instructions as hex strings."""
        return [format(int(instr, 2), '06X') for instr in self.instructions]

    def get_instruction_stats(self):
        """Return statistics about generated instructions."""
        stats = {
            'total_instructions': len(self.instructions),
            'prog_instructions': sum(1 for instr in self.instructions if int(instr[:8][-2:], 2) == self.instruction_types['PROG']),
            'exe_instructions': sum(1 for instr in self.instructions if int(instr[:8][-2:], 2) == self.instruction_types['EXE']),
            'end_instructions': sum(1 for instr in self.instructions if int(instr[:8][-2:], 2) == self.instruction_types['END']),
            'memory_reads': sum(1 for instr in self.instructions if instr[14:16] in ['10', '11']),
            'memory_writes': sum(1 for instr in self.instructions if instr[14:16] == '01'),
            'mac_operations': self.mac_operations
        }
        return stats
