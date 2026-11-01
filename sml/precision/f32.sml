(* Matrix storage is selected at build time; normalization weights stay f32. *)
structure Matrices = struct
  val precision = "f32"
  structure Array = Real32Array
  structure Slice = Real32ArraySlice
  structure A2 = qwen.Real32Array2
  structure A3 = qwen.Real32Array3
  structure A4 = qwen.Real32Array4
  fun fromReal32 (x:Real32.real) = x
  fun toReal32 (x:Real32.real) = x
  fun config () = qwen.Config.cache
      (if qwen.backend="cuda" then SOME "qwen-cuda-f32.cache" else NONE) qwen.Config.default
end
