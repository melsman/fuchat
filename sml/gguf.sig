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
  (* Destination indices implement reshape/transpose. f16 reads retain source
     half bits; f32 source tensors are rounded to nearest when narrowed. *)
  val readTensor32Into : model -> reader -> tensor ->
      (int * Real32.real -> unit) -> (int -> int) -> unit
  val readTensor16Into : model -> reader -> tensor ->
      (int * Real16.real -> unit) -> (int -> int) -> unit
end
