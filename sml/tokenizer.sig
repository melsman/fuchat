(* NFC-normalized Qwen byte-level BPE; strings contain UTF-8 bytes. *)
signature TOKENIZER = sig
  type t
  val create : Gguf.model -> t
  val tokenId : t -> string -> int
  val encode : t -> string -> int vector
  (* raw can end in an incomplete UTF-8 sequence when decoding a batch. *)
  val raw : t -> int vector -> string
  val decode : t -> int vector -> string
end
