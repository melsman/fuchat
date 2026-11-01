structure Options : OPTIONS = struct
  datatype mode = Chat | Help | Weights | Trace of string | Encode of string
                | Decode of string | TokenizerTest of string | RenderTest of string
                | Benchmark of string
  type t = {modelFile:string, context:int, count:int, limit:int,
            thinking:bool, tools:bool, maxToolRounds:int, system:string,
            prompt:string option, mode:mode, benchmarkCount:int, benchmarkRuns:int}
  val usage = "Usage: build/chat-sml [--model FILE] [--cs N] [--cnt N] [--max-new-tokens N]\n\
              \                      [--system TEXT] [--prompt TEXT] [--think] [--tools] [--max-tool-rounds N]\n\
              \       build/chat-sml --model FILE --encode TEXT_FILE | --decode TOKEN_IDS\n\
              \       build/chat-sml --model FILE --benchmark IDS_FILE [--benchmark-tokens N] [--benchmark-runs N]\n\
              \Local unquantized Qwen3 GGUF, build-selected f32/f16 inference.\n\
              \Commands: quit/clear/think/nothink/show/hide.\n"
  fun parse defaultContext args : t =
      let
        fun positive text =
            case Int.scan StringCvt.DEC Substring.getc (Substring.full text) of
              SOME (n,rest) => if Substring.isEmpty rest andalso n>0 then n
                               else Util.fail ("expected a positive integer: " ^ text)
            | NONE => Util.fail ("expected a positive integer: " ^ text)
        val modelFile = ref "Qwen3-0.6B-f16.gguf"
        val context = ref defaultContext
        val count = ref 5
        val limit = ref 8192
        val thinking = ref false
        val tools = ref false
        val maxToolRounds = ref 8
        val benchmarkCount = ref 32
        val benchmarkRuns = ref 3
        val system = ref ""
        val prompt = ref (NONE:string option)
        val mode = ref Chat
        fun arguments [] = ()
          | arguments ("--help"::_) = mode:=Help
          | arguments ("--model"::s::rest) = (modelFile:=s; arguments rest)
          | arguments ("--cs"::s::rest) = (context:=positive s; arguments rest)
          | arguments ("--cnt"::s::rest) = (count:=positive s; arguments rest)
          | arguments ("--max-new-tokens"::s::rest) = (limit:=positive s; arguments rest)
          | arguments ("--think"::rest) = (thinking:=true; arguments rest)
          | arguments ("--tools"::rest) = (tools:=true; arguments rest)
          | arguments ("--max-tool-rounds"::s::rest) = (maxToolRounds:=positive s; arguments rest)
          | arguments ("--system"::s::rest) = (system:=s; arguments rest)
          | arguments ("--prompt"::s::rest) = (prompt:=SOME s; arguments rest)
          | arguments ("--benchmark"::s::rest) = (mode:=Benchmark s; arguments rest)
          | arguments ("--benchmark-tokens"::s::rest) = (benchmarkCount:=positive s; arguments rest)
          | arguments ("--benchmark-runs"::s::rest) = (benchmarkRuns:=positive s; arguments rest)
          | arguments ("--weights"::rest) = (mode:=Weights; arguments rest)
          | arguments ("--encode"::s::rest) = (mode:=Encode s; arguments rest)
          | arguments ("--decode"::s::rest) = (mode:=Decode s; arguments rest)
          | arguments ("--trace"::s::rest) = (mode:=Trace s; arguments rest)
          | arguments ("--tokenizer-test"::s::rest) = (mode:=TokenizerTest s; arguments rest)
          | arguments ("--render-test"::s::rest) = (mode:=RenderTest s; arguments rest)
          | arguments (s::_) = Util.fail ("unknown or incomplete argument: " ^ s)
        val () = arguments args
      in {modelFile= !modelFile,context= !context,count= !count,limit= !limit,
          thinking= !thinking,tools= !tools,maxToolRounds= !maxToolRounds,
          system= !system,prompt= !prompt,mode= !mode,
          benchmarkCount= !benchmarkCount,benchmarkRuns= !benchmarkRuns}
      end
end
