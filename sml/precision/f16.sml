(* The f16 engine keeps normalization weights in f32, as specified in Futhark. *)
structure Matrices = struct
  val precision = "f16"
  structure Array = Real16Array
  structure Slice = Real16ArraySlice
  structure A2 = qwen.Real16Array2
  structure A3 = qwen.Real16Array3
  structure A4 = qwen.Real16Array4
  fun fromReal32 x = Real16.fromLarge IEEEReal.TO_NEAREST (Real32.toLarge x)
  fun toReal32 x = Real32.fromLarge IEEEReal.TO_NEAREST (Real16.toLarge x)
  fun config () = qwen.Config.cache
      (if qwen.backend="cuda" then SOME "qwen-cuda-f16.cache" else NONE) qwen.Config.default
end
