; ModuleID = "matrix_multiply"
target triple = "unknown-unknown-unknown"
target datalayout = ""

define void @"matrixMultiply"([3 x [3 x i32]]* %"A", [3 x [3 x i32]]* %"B", [3 x [3 x i32]]* %"C")
{
entry:
  %"i" = alloca i32
  %"j" = alloca i32
  %"k" = alloca i32
  store i32 0, i32* %"i"
  br label %"loop_i_cond"
loop_i_cond:
  %".7" = load i32, i32* %"i"
  %".8" = icmp slt i32 %".7", 3
  br i1 %".8", label %"loop_i_body", label %"loop_i_end"
loop_i_body:
  store i32 0, i32* %"j"
  br label %"loop_j_cond"
loop_i_inc:
  %".47" = load i32, i32* %"i"
  %".48" = add i32 %".47", 1
  store i32 %".48", i32* %"i"
  br label %"loop_i_cond"
loop_i_end:
  ret void
loop_j_cond:
  %".12" = load i32, i32* %"j"
  %".13" = icmp slt i32 %".12", 3
  br i1 %".13", label %"loop_j_body", label %"loop_j_end"
loop_j_body:
  %".15" = load i32, i32* %"i"
  %".16" = load i32, i32* %"j"
  %".17" = getelementptr [3 x [3 x i32]], [3 x [3 x i32]]* %"C", i32 0, i32 %".15", i32 %".16"
  store i32 0, i32* %".17"
  store i32 0, i32* %"k"
  br label %"loop_k_cond"
loop_j_inc:
  %".42" = load i32, i32* %"j"
  %".43" = add i32 %".42", 1
  store i32 %".43", i32* %"j"
  br label %"loop_j_cond"
loop_j_end:
  br label %"loop_i_inc"
loop_k_cond:
  %".21" = load i32, i32* %"k"
  %".22" = icmp slt i32 %".21", 3
  br i1 %".22", label %"loop_k_body", label %"loop_k_end"
loop_k_body:
  %".24" = load i32, i32* %"i"
  %".25" = load i32, i32* %"j"
  %".26" = load i32, i32* %"k"
  %".27" = getelementptr [3 x [3 x i32]], [3 x [3 x i32]]* %"A", i32 0, i32 %".24", i32 %".26"
  %".28" = load i32, i32* %".27"
  %".29" = getelementptr [3 x [3 x i32]], [3 x [3 x i32]]* %"B", i32 0, i32 %".26", i32 %".25"
  %".30" = load i32, i32* %".29"
  %".31" = mul i32 %".28", %".30"
  %".32" = getelementptr [3 x [3 x i32]], [3 x [3 x i32]]* %"C", i32 0, i32 %".24", i32 %".25"
  %".33" = load i32, i32* %".32"
  %".34" = add i32 %".33", %".31"
  store i32 %".34", i32* %".32"
  br label %"loop_k_inc"
loop_k_inc:
  %".37" = load i32, i32* %"k"
  %".38" = add i32 %".37", 1
  store i32 %".38", i32* %"k"
  br label %"loop_k_cond"
loop_k_end:
  br label %"loop_j_inc"
}
