(* LIFO ownership of independently releasable resources. *)
signature RESOURCE = sig
  type t
  val create : unit -> t
  val defer : t -> (unit -> unit) -> unit
  val own : t -> ('a -> unit) -> 'a -> 'a
  (* Idempotent. Attempts every cleanup and raises the first cleanup failure. *)
  val release : t -> unit
  (* Retains resources on success; releases them if the action raises. *)
  val onFailure : t -> (unit -> 'a) -> 'a
  (* Always releases resources. Preserves an exception raised by the action. *)
  val scope : (t -> 'a) -> 'a
end
