(* Compare hosts using fixed token IDs and one decode token per entry call.
   A prefill measurement includes the first generated token. *)
structure Benchmark = struct
  fun realNumber value = Json.Number
      (String.translate (fn #"~" => "-" | c => String.str c) (Real.toString value))
  fun run (engine:Engine.t) input count runs loadSeconds =
      let
        fun one () =
            let
              val () = Engine.clear engine
              val start = Time.now ()
              val first = Engine.generate engine input 1
              val promptSeconds = Time.toReal (Time.-(Time.now (),start))
              val decodeStart = Time.now ()
              fun loop (ids,generated,last) =
                  if generated>=count orelse Vector.length ids> #size engine orelse last= #eos engine then ids
                  else let val next = Engine.generate engine ids 1
                           val out = #newTokens next
                       in if Vector.length out=0 then #ids next
                          else loop(#ids next,generated+Vector.length out,Vector.sub(out,Vector.length out-1)) end
              val out = #newTokens first
              val all = if Vector.length out=0 then #ids first
                        else loop(#ids first,Vector.length out,Vector.sub(out,Vector.length out-1))
              val seconds = Time.toReal (Time.-(Time.now (),decodeStart))
              val tokens = Vector.tabulate(Vector.length all-Vector.length input,fn i => Vector.sub(all,Vector.length input+i))
            in Json.Object [("prompt_seconds",realNumber promptSeconds),("generation_seconds",realNumber seconds),
                 ("prompt_tokens",Json.Number(Int.toString(Vector.length input))),
                 ("generated_tokens",Json.Number(Int.toString(Vector.length tokens))),
                 ("ids",Json.Array (Vector.foldr (fn (id,acc) => Json.Number(Int.toString id)::acc) [] tokens))] end
        val () = ignore (one ())
        val results = List.tabulate(runs,fn _ => one ())
      in Json.Object [("backend",Json.String qwen.backend),("precision",Json.String Matrices.precision),
                       ("load_seconds",realNumber loadSeconds),("context",Json.Number(Int.toString (#size engine))),
                       ("warmups",Json.Number "1"),("runs",Json.Array results)] end
end
