structure Main = struct
  fun run arguments =
      let
        val modelFile = ref "Qwen3-0.6B-f16.gguf"
        val context = ref 8192
        val count = ref 5
        val limit = ref 8192
        val thinking = ref false
        val tools = ref false
        val maxToolRounds = ref 8
        val system = ref ""
        val prompt = ref (NONE:string option)
        val mode = ref "chat"
        val fixture = ref ""
        fun help () =
            (print "Usage: build/chat-sml [--model FILE] [--cs N] [--cnt N] [--max-new-tokens N]\n\
                   \                      [--system TEXT] [--prompt TEXT] [--think] [--tools] [--max-tool-rounds N]\n\
                   \       build/chat-sml --model FILE --encode TEXT_FILE\n\
                   \       build/chat-sml --model FILE --decode TOKEN_IDS\n\
                   \Local unquantized Qwen3 GGUF, f32 inference. Commands: quit/clear/think/nothink/show/hide.\n";
             OS.Process.exit OS.Process.success)
        fun parse [] = ()
          | parse ("--help"::_) = help ()
          | parse ("--model"::s::rest) = (modelFile:=s; parse rest)
          | parse ("--cs"::s::rest) = (context:=Util.int s; parse rest)
          | parse ("--cnt"::s::rest) = (count:=Util.int s; parse rest)
          | parse ("--max-new-tokens"::s::rest) = (limit:=Util.int s; parse rest)
          | parse ("--think"::rest) = (thinking:=true; parse rest)
          | parse ("--tools"::rest) = (tools:=true; parse rest)
          | parse ("--max-tool-rounds"::s::rest) = (maxToolRounds:=Util.int s; parse rest)
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
        val () = if !context>0 andalso !count>0 andalso !limit>0 andalso !maxToolRounds>0 then () else Util.fail "counts and context size must be positive"
        fun renderTests () = Protocol.lines (!fixture) (fn line =>
            let
              val rendered =
                  if String.isPrefix "{" line then
                    let val test = Json.fromString line
                        val messages = List.map Chat.fromJson (Json.array (Json.get test "messages"))
                        val thinking = case Json.get test "thinking" of Json.Bool b => b | _ => false
                        val generation = case Json.get test "generation" of Json.Bool b => b | _ => false
                        val tools = case Json.get test "tools" of Json.Array tools => tools | _ => []
                    in Chat.renderWith {messages=messages,generation=generation,thinking=thinking,tools=tools} end
                  else
                    let val fields = String.fields (fn c => c = #"\t") line
                        val generation = hd fields = "1"
                        fun message field = case String.fields (fn c => c = #":") field of
                                              [role,content] => Chat.simple role (Protocol.unhex content)
                                            | _ => Util.fail "invalid render fixture"
                    in Chat.render (List.map message (tl fields)) generation end
            in print (Protocol.hex rendered ^ "\n") end)
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
              val initial = if !system="" then [] else [Chat.simple "system" (!system)]
              val messages = ref initial
              val show = ref false
              val () = if !tools then List.app (fn (entry:ToolEntries.entry) =>
                          Util.log ("Tool " ^ #name entry ^ ": " ^ #description entry)) ToolEntries.all else ()
              fun emit text = (print text; TextIO.flushOut TextIO.stdOut)
              fun renderCurrent () = Chat.renderWith {messages= !messages,generation=false,thinking= !thinking,
                                                       tools=if !tools then Tools.descriptions else []}
              fun respond input =
                  let
                    val started = Time.now ()
                    val options = {thinking= !thinking,tools= !tools,show= !show,count= !count,limit= !limit,
                                   context= !context,maxToolRounds= !maxToolRounds}
                    fun generate ids amount =
                        let val result = Engine.generate engine ids amount
                        in {ids= #ids result,newTokens= #newTokens result,reused= #reused result,eos= #eos engine} end
                    fun execute (call:ToolCall.t) =
                        let val result = Tools.run (!tools) (#ctx engine) call
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
            in case !prompt of SOME input => respond input
                             | NONE => (emit "Qwen3 / MLKit. Commands: quit, clear, think, nothink, show, hide.\n"; loop ())
            end
      in if !mode="render-test" then renderTests () else withModel ()
      end
end
val () = Main.run (CommandLine.arguments ())
         handle ex => (Util.log ("fuchat: " ^ General.exnMessage ex); OS.Process.exit OS.Process.failure)
