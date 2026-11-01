(* Matrix storage is selected at build time; normalization weights stay f32. *)
structure Matrices = struct
  val precision = "f32"
  structure Array = Real32Array
  structure Slice = Real32ArraySlice
  structure A2 = qwen.Real32Array2
  structure A3 = qwen.Real32Array3
  structure A4 = qwen.Real32Array4
  val zero = Real32.fromInt 0
  val readRows = Gguf.readTensor32Rows
  fun fill array (start,count,get) =
      Slice.modifyi (fn (i,_) => get i) (Slice.slice(array,start,SOME count))
  fun toReal32 (x:Real32.real) = x
  fun config () = qwen.Config.cache
      (if qwen.backend="cuda" then SOME "qwen-cuda-f32.cache" else NONE) qwen.Config.default
end
