# pPIM Compiler - LLVM Approach

## Overview

This project implements a compiler for a custom Processor-in-Memory (PIM) architecture tailored for AI/ML workloads. The compiler translates C++ matrix multiplication programs with parameterized dimensions into a stream of custom ISA-compatible instructions that efficiently leverage the PIM's unique computational capabilities.

The LLVM-based implementation utilizes LLVM's intermediate representation for optimization and generates PIM-specific instructions that follow the 24-bit fixed-length instruction format specified in the architecture.

## Features

- Parses C++ matrix multiplication code to extract matrix dimensions
- Generates LLVM IR for matrix multiplication operations
- Creates optimized memory maps for matrices across memory banks
- Produces PIM-specific instructions in both binary and human-readable formats
- Implements block-based processing for efficient memory access
- Provides detailed statistics about generated instructions

## Requirements

- Python 3.6+
- llvmlite
- pycparser

Install dependencies using:
```bash
pip install llvmlite pycparser
```

## Project Structure

```
PIM_Compiler/
├── matrix_parser.py     # Extracts matrix dimensions from C++ code
├── llvm_generator.py    # Generates LLVM IR for matrix multiplication
├── memory_mapper.py     # Optimizes matrix placement across memory banks
├── instruction_generator.py  # Translates operations into PIM instructions
├── output_handler.py    # Produces output files in various formats
├── pim_compiler.py      # Main compiler script
└── sample.cpp           # Example matrix multiplication code
```

## Usage

```bash
python pim_compiler.py  [--output-dir OUTPUT_DIR]
```

### Arguments:
- `cpp_file_path`: Path to the C++ source file containing matrix multiplication code
- `--output-dir`: Optional output directory (default: 'PIM_Compiler')

### Example:
```bash
python pim_compiler.py sample.cpp --output-dir my_output
```

## Output Files

The compiler generates several output files in the specified directory:

1. `binary_instructions.txt`: Binary representation of pPIM instructions
2. `readable_instructions.txt`: Human-readable format of instructions
3. `memory_map.txt`: Memory mapping information for matrices
4. `matrix_multiply.ll`: LLVM IR representation
5. `instruction_stats.txt`: Statistics about generated instructions
6. `binary_instructions_hex.txt`: Hexadecimal representation of instructions

## Instruction Format

The pPIM ISA features a fixed-length 24-bit instruction format with:

- 8-bit operation segment (6-bit pointer + 2-bit opcode)
- 6 reserved bits
- 10-bit memory access segment (1-bit read, 1-bit write, 8-bit row address)

Three primary instruction types are supported:
- `PROG` (opcode 1): Programs LUT cores for specific operations
- `EXE` (opcode 2): Executes operations using configured cores
- `END` (opcode 3): Resets program counters and registers

## Performance

The LLVM-based implementation provides excellent optimization capabilities and extensibility, though it generates more instructions compared to the direct implementation approach. Performance metrics show that the efficiency ratio (MAC operations / total instructions) improves with larger matrices.

## License

This project is provided for educational and research purposes.

## Acknowledgments

Based on the paper "Flexible Instruction Set Architecture for Programmable Look-up Table based Processing-in-Memory."
