(* Report hidden work without printing reasoning or disturbing stdout. The
   clock and sink are injected so throttling can be checked deterministically. *)
structure Progress = struct
  datatype phase = Prompt of int | Thinking | Tool | Answer | Done
  type t = {emit:string -> unit, elapsed:unit -> real, last:phase option ref, at:real ref}
  fun create emit elapsed : t = {emit=emit,elapsed=elapsed,last=ref NONE,at=ref 0.0}
  fun update ({emit,elapsed,last,at}:t) (phase,tokens) =
      let
        val now = elapsed ()
        fun report text = (emit text; last:=SOME phase; at:=now)
        fun hidden label =
            if !last<>SOME phase orelse now- !at >= 5.0 then
              report (label ^ ": " ^ Int.toString tokens ^ " tokens (" ^
                      Real.fmt (StringCvt.FIX (SOME 1)) now ^ "s elapsed)")
            else ()
      in
        case phase of
          Prompt count => report ("Processing prompt (" ^ Int.toString count ^ " tokens)...")
        | Thinking => hidden "Thinking"
        | Tool => hidden "Preparing tool call"
        | Answer => last:=SOME Answer
        | Done => last:=NONE
      end
end
