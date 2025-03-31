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