structure Main = struct
  fun run arguments =
      let
        val modelFile = ref "Qwen3-0.6B-f16.gguf"
        val context = ref 8192
        val count = ref 5
        val limit = ref 8192
        val system = ref ""
        val prompt = ref (NONE:string option)
        val mode = ref "chat"
        val fixture = ref ""
        fun help () =
            (print "Usage: build/chat-sml [--model FILE] [--cs N] [--cnt N] [--max-new-tokens N]\n\
                   \                      [--system TEXT] [--prompt TEXT]\n\
                   \       build/chat-sml --model FILE --encode TEXT_FILE\n\
                   \       build/chat-sml --model FILE --decode TOKEN_IDS\n\
                   \Local unquantized Qwen3 GGUF, f32 inference. Commands: quit/clear/show/hide.\n";
             OS.Process.exit OS.Process.success)
        fun parse [] = ()
          | parse ("--help"::_) = help ()
          | parse ("--model"::s::rest) = (modelFile:=s; parse rest)
          | parse ("--cs"::s::rest) = (context:=Util.int s; parse rest)
          | parse ("--cnt"::s::rest) = (count:=Util.int s; parse rest)
          | parse ("--max-new-tokens"::s::rest) = (limit:=Util.int s; parse rest)
          | parse ("--system"::s::rest) = (system:=s; parse rest)
          | parse ("--prompt"::s::rest) = (prompt:=SOME s; parse rest)
          | parse ("--weights"::rest) = (mode:="weights"; parse rest)
          | parse ("--encode"::s::rest) = (mode:="encode"; fixture:=s; parse rest)
          | parse ("--decode"::s::rest) = (mode:="decode"; fixture:=s; parse rest)
          | parse ("--trace"::s::rest) = (mode:="trace"; fixture:=s; parse rest)
          | parse ("--tokenizer-test"::s::rest) = (mode:="tokenizer-test"; fixture:=s; parse rest)
          | parse ("--render-test"::s::rest) = (mode:="render-test"; fixture:=s; parse rest)
          | parse (s::_) = Util.fail ("unknown or incomplete argument: " ^ s)
        val () = parse arguments
        val () = if !context>0 andalso !count>0 andalso !limit>0 then () else Util.fail "counts and context size must be positive"
        fun renderTests () = Protocol.lines (!fixture) (fn line =>
            let
              val fields = String.fields (fn c => c = #"\t") line
              val generation = hd fields = "1"
              fun message field = case String.fields (fn c => c = #":") field of
                                    [role,content] => {role=role,content=Protocol.unhex content}
                                  | _ => Util.fail "invalid render fixture"
            in print (Protocol.hex (Chat.render (List.map message (tl fields)) generation) ^ "\n") end)
        fun withModel () =
            let
              val model = Gguf.load (!modelFile)
              fun withTokenizer () =
                  let val tokenizer = Tokenizer.create model
                  in
                    case !mode of
                      "encode" => print (Protocol.ids (Tokenizer.encode tokenizer (Util.readFile (!fixture))) ^ "\n")
                    | "decode" => print (Tokenizer.decode tokenizer (Protocol.parseIds (!fixture)))
                    | "tokenizer-test" => Protocol.lines (!fixture) (fn line =>
                        let val text = Protocol.unhex line
                            val ids = Tokenizer.encode tokenizer text
                            val pieces = PreTokenizer.split (UnicodeNormalise.nfc text)
                        in print (Protocol.ids ids ^ "\t" ^ Protocol.hex (Tokenizer.decode tokenizer ids)
                                  ^ "\t" ^ String.concatWith "," (List.map Protocol.hex pieces) ^ "\n") end)
                    | _ => withEngine model (SOME tokenizer)
                  end
            in if !mode="weights" orelse !mode="trace" then withEngine model NONE else withTokenizer ()
            end
        and withEngine model tokenizer =
            let
              val engine = Engine.openModel model (!context)
              fun trace line =
                  if line="clear" then Engine.clear engine
                  else case String.fields (fn c => c = #"\t") line of
                    [amount,ids] =>
                      let val result = Engine.generate engine (Protocol.parseIds ids) (Util.int amount)
                      in print (Int.toString (#reused result) ^ "\t" ^ Protocol.ids (#newTokens result) ^ "\n") end
                  | _ => Util.fail "invalid generation trace"
              fun action () =
                  if !mode="weights" then Inspect.weights (#params engine)
                  else if !mode="trace" then Protocol.lines (!fixture) trace
                  else interactive engine (valOf tokenizer)
            in Util.finally action (fn () => Engine.close engine)
            end
        and interactive engine tokenizer =
            let
              val initial = if !system="" then [] else [{role="system",content= !system}]
              val messages = ref initial
              val show = ref false
              fun respond input =
                  let
                    val proposed = !messages @ [{role="user",content=input}]
                    val text = Chat.render proposed true
                    val ids = Tokenizer.encode tokenizer text
                  in
                    if Vector.length ids > !context then Util.log "Context is full; use clear or a shorter message."
                    else
                      let
                        val () = if !show then (print (text ^ "\n"); TextIO.flushOut TextIO.stdOut) else ()
                        val pending = ref ""
                        val captured = ref ([]:int vector list)
                        val total = ref 0
                        val started = Time.now ()
                        fun stream tokens =
                            let val text = !pending ^ Tokenizer.raw tokenizer tokens
                                val (ready,tail) = Unicode.clean false text
                            in pending:=tail; print ready; TextIO.flushOut TextIO.stdOut end
                        fun generate (ids,first) =
                            if !total >= !limit orelse Vector.length ids > !context then ()
                            else
                              let
                                val result = Engine.generate engine ids (Int.min (if first then 1 else !count,!limit- !total))
                                val new = #newTokens result
                                val n = Vector.length new
                                val eos = n>0 andalso Vector.sub(new,n-1)= #eos engine
                                val visible = if eos then Vector.tabulate(n-1,fn i => Vector.sub(new,i)) else new
                                val () = if first then Util.log ("Reused " ^ Int.toString (#reused result) ^ " cached tokens") else ()
                                val () = captured := visible :: !captured
                                val () = total := !total+n
                                val () = stream visible
                              in if eos orelse n=0 then () else generate (#ids result,false)
                              end
                        val () = generate(ids,true)
                        val () = print (#1 (Unicode.clean true (!pending)) ^ "\n")
                        val answer = Unicode.trim (Tokenizer.decode tokenizer (Vector.concat (List.rev (!captured))))
                        val () = messages := proposed @ [{role="assistant",content=answer}]
                        val elapsed = Time.toReal (Time.-(Time.now (),started))
                      in Util.log (Int.toString (!total) ^ " generated tokens in " ^ Real.toString elapsed ^ "s")
                      end
                  end
              fun loop () =
                  (print "chat> "; TextIO.flushOut TextIO.stdOut;
                   case TextIO.inputLine TextIO.stdIn of
                     NONE => ()
                   | SOME line =>
                     case Util.stripNewline line of
                       "quit" => ()
                     | "clear" => (messages:=initial; Engine.clear engine; print "Context cleared\n"; loop ())
                     | "show" => (show:=true; print (Chat.render (!messages) false ^ "\n"); loop ())
                     | "hide" => (show:=false; loop ())
                     | "" => loop ()
                     | input => (respond input; loop ()))
            in case !prompt of SOME input => respond input
                             | NONE => (print "Qwen3 / MLKit. Commands: quit, clear, show, hide.\n"; loop ())
            end
      in if !mode="render-test" then renderTests () else withModel ()
      end
end
val () = Main.run (CommandLine.arguments ())
         handle ex => (Util.log ("fuchat: " ^ General.exnMessage ex); OS.Process.exit OS.Process.failure)
