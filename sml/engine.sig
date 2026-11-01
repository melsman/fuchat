(* Owns one context, model parameters, and a replaceable KV-cache handle. *)
signature ENGINE = sig
  type t
  type result = {ids:int vector, newTokens:int vector, reused:int}
  (* Always closes the engine, including when the callback raises. *)
  val withModel : Gguf.model -> int -> (t -> 'a) -> 'a
  val capacity : t -> int
  val eos : t -> int
  val clear : t -> unit
  val prefix : int vector -> int vector -> int
  (* Borrows input IDs; replaces the owned cache only after successful output
     projection. On inference failure the engine is closed and must not be reused. *)
  val generate : t -> int vector -> int -> result
  (* Borrowed handles for tools and diagnostics. Do not free or retain them. *)
  val useContext : t -> (qwen.ctx -> 'a) -> 'a
  val useParams : t -> (qwen.Opaque.Params.t -> 'a) -> 'a
end
