structure ChatApp = struct
  fun run (options:Options.t) = Host.withModel options (fn model =>
      let val tokenizer = Tokenizer.create model
      in Host.withEngine options model (fn (engine,_) => interactive options engine tokenizer) end)
  and interactive (options:Options.t) engine tokenizer =
      let
        val {system,thinking=initialThinking,tools,count,limit,maxToolRounds,prompt,...} = options
        val thinking = ref initialThinking
        val initial = if system="" then [] else [Chat.simple "system" (system)]
        val messages = ref initial
        val show = ref false
        val () = if tools then List.app (fn (entry:ToolEntries.entry) =>
                    Util.log ("Tool " ^ #name entry ^ ": " ^ #description entry)) ToolEntries.all else ()
        fun emit text = (print text; TextIO.flushOut TextIO.stdOut)
        fun renderCurrent () = Chat.renderWith {messages= !messages,generation=false,thinking= !thinking,
                                                 tools=if tools then Tools.descriptions else []}
        fun respond input =
            let
              val started = Time.now ()
              val options = {thinking= !thinking,tools= tools,show= !show,count= count,limit= limit,
                             context=Engine.capacity engine,maxToolRounds= maxToolRounds}
              fun generate ids amount =
                  let val result = Engine.generate engine ids amount
                  in {ids= #ids result,newTokens= #newTokens result,reused= #reused result,eos= Engine.eos engine} end
              fun execute (call:ToolCall.t) =
                  let val result = Engine.useContext engine (fn ctx => Tools.run tools ctx call)
                  in Util.log ("Tool " ^ Json.quote (#name call) ^ ": " ^ result); result end
              val progress = Progress.create Util.log (fn () => Time.toReal (Time.-(Time.now (),started)))
              val result = Session.reply tokenizer options generate execute emit Util.log (Progress.update progress)
                                         (!messages @ [Chat.simple "user" input])
              val () = if #total result=0 then () else messages:= #messages result
              val elapsed = Time.toReal (Time.-(Time.now (),started))
            in if #total result=0 then () else
                 (Util.log ("Reused " ^ Int.toString (#reused result) ^ " cached tokens");
                  Util.log (Int.toString (#total result) ^ " generated tokens in " ^ Real.toString elapsed ^ "s")) end
        fun loop () =
            (emit "chat> ";
             case TextIO.inputLine TextIO.stdIn of
               NONE => ()
             | SOME line =>
               case Util.stripNewline line of
                 "quit" => ()
               | "clear" => (messages:=initial; Engine.clear engine; emit "Context cleared\n"; loop ())
               | "show" => (show:=true; emit (renderCurrent () ^ "\n"); loop ())
               | "hide" => (show:=false; loop ())
               | "think" => (thinking:=true; emit "Thinking mode enabled\n"; loop ())
               | "nothink" => (thinking:=false; emit "Thinking mode disabled\n"; loop ())
               | "" => loop ()
               | input => (respond input; loop ()))
      in case prompt of SOME input => respond input
                       | NONE => (emit "Qwen3 / MLKit. Commands: quit, clear, think, nothink, show, hide.\n"; loop ())
      end
end
