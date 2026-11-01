(* LIFO ownership scopes. Cleanup attempts every release, even if one fails.
   On an action failure, preserve that exception rather than a cleanup error. *)
structure Resource :> RESOURCE = struct
  type t = (unit -> unit) list ref
  fun create () : t = ref []
  fun defer (scope:t) cleanup = scope := cleanup :: !scope
  fun own scope free value = (defer scope (fn () => free value); value)
  fun release (scope:t) =
      let val pending = !scope
          val () = scope := []
          fun loop ([],NONE) = ()
            | loop ([],SOME ex) = raise ex
            | loop (free::rest,error) =
                let val error = (free (); error) handle ex =>
                                  (case error of NONE => SOME ex | SOME _ => error)
                in loop(rest,error) end
      in loop(pending,NONE) end
  fun onFailure scope action =
      action () handle ex => ((release scope handle _ => ()); raise ex)
  fun scope action =
      let val resources = create ()
          val result = onFailure resources (fn () => action resources)
      in release resources; result end
end
