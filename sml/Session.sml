(* The same orchestration runs with Engine.generate in production and with
   scripted token batches in tests; dispatch is limited to the typed registry. *)
structure Session = struct
  type options = {thinking:bool, tools:bool, show:bool, count:int, limit:int, context:int, maxToolRounds:int}
  type generation = {ids:int vector, newTokens:int vector, reused:int, eos:int}
  fun reply tokenizer (options:options) (generate:int vector -> int -> generation) execute emit warn progress messages =
      let
        val remaining = ref (#limit options)
        val total = ref 0
        val reusedFirst = ref (NONE:int option)
        fun round history toolRounds =
            if !remaining<=0 then (warn "Generation limit reached"; history)
            else
              let
                val text = Chat.renderWith {messages=history,generation=true,thinking= #thinking options,
                                             tools=if #tools options then Tools.descriptions else []}
                val ids = Tokenizer.encode tokenizer text
              in
                if Vector.length ids > #context options then (warn "Context is full; use clear or a shorter message."; history)
                else
                  let
                    val () = progress (Progress.Prompt (Vector.length ids),!total)
                    val () = if #show options then emit (text ^ "\n") else ()
                    val capture = Capture.create tokenizer
                    val pending = ref ""
                    fun stream bytes =
                        let val (ready,tail) = Unicode.clean false (!pending ^ bytes)
                        in pending:=tail; emit ready end
                    fun step (ids,first) =
                        if !remaining<=0 orelse Vector.length ids > #context options then false
                        else
                          let
                            val amount = Int.min (if first then 1 else #count options,!remaining)
                            val result = generate ids amount
                            val new = #newTokens result
                            val n = Vector.length new
                            val () = if first then
                              (if Option.isSome (!reusedFirst) then () else reusedFirst:=SOME (#reused result)) else ()
                            val () = total:= !total+n
                            val () = remaining:= !remaining-n
                            val eos = ref false
                            fun token id =
                                if id= #eos result then eos:=true
                                else let val normal = Capture.feed capture id
                                     in stream (if #show options then Tokenizer.raw tokenizer (Vector.fromList [id]) else normal) end
                            val () = Vector.app token new
                            val phase = case !(#mode capture) of Capture.Reasoning => Progress.Thinking
                                          | Capture.Tool => Progress.Tool | Capture.Answer => Progress.Answer
                            val () = if #show options then () else progress (phase,!total)
                          in if !eos then true else if n=0 then false else step(#ids result,false) end
                    val complete = step(ids,true)
                    val () = emit (#1 (Unicode.clean true (!pending)) ^ "\n")
                    val parsed = Capture.finish capture complete
                    val () = List.app warn (#warnings parsed)
                    val assistant = #message parsed
                    val calls = #calls assistant
                    val history = history @ [assistant]
                  in
                    if List.null calls then history
                    else
                      let
                        val capped = toolRounds>= #maxToolRounds options
                        fun dispatch call =
                            Chat.simple "tool" (if capped then Tools.error "Tool round limit reached" else execute call)
                        val results = List.map dispatch calls
                        val next = history @ results
                      in if capped then (warn "Tool round limit reached"; next) else round next (toolRounds+1) end
                  end
              end
        val result = round messages 0
        val () = progress (Progress.Done,!total)
      in {messages=result,total= !total,reused=Option.getOpt(!reusedFirst,0)} end
end
