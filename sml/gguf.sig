(* Local little-endian GGUF v2/v3 metadata and bounded tensor reads. *)
signature GGUF = sig
  type model
  type reader
  type tensor = {name:string, shape:int list, kind:int, offset:int}
  val load : string -> model
  val integer : model -> string -> int
  val string : model -> string -> string
  val strings : model -> string -> string vector
  val integers : model -> string -> int vector
  val tensor : model -> string -> tensor
  (* The reader is borrowed for the callback and always closed afterwards. *)
  val withReader : model -> (reader -> 'a) -> 'a
  (* Rows arrive in ascending order. The callback receives first row, row
     count, and a decoder indexed from 0 through count*columns-1. The decoder
     borrows a reusable buffer and must not escape the callback. f16 source
     bits are retained; narrowing f32 uses round-to-nearest. The reader
     reuses a 64 KiB input buffer; a wider row uses one additional row buffer. Short reads
     fill the same buffer. *)
  val readTensor32Rows : model -> reader -> tensor -> int ->
      (int * int * (int -> Real32.real) -> unit) -> unit
  val readTensor16Rows : model -> reader -> tensor -> int ->
      (int * int * (int -> Real16.real) -> unit) -> unit
end
