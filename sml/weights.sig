(* Qwen3 shape validation and preparation for the selected Futhark precision. *)
signature WEIGHTS = sig
  type dimensions = {blocks:int, vocab:int, embedding:int, feedForward:int,
                     heads:int, kvHeads:int, headDim:int}
  val dimensions : Gguf.model -> dimensions
  (* Borrows the context. The caller owns the returned Params handle;
     intermediate array handles and the GGUF reader are scoped internally. *)
  val load : qwen.ctx -> Gguf.model -> qwen.Opaque.Params.t
end
