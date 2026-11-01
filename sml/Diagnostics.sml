(* Machine-readable inspection and regression protocols live outside ChatApp. *)
structure Diagnostics = struct
  fun renderTests fixture = Protocol.lines fixture (fn line =>
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

  fun trace engine fixture = Protocol.lines fixture (fn line =>
      if line="clear" then Engine.clear engine
      else case String.fields (fn c => c = #"\t") line of
        [amount,ids] =>
          let val result = Engine.generate engine (Protocol.parseIds ids) (Util.int amount)
          in print (Int.toString (#reused result) ^ "\t" ^ Protocol.ids (#newTokens result) ^ "\n") end
      | _ => Util.fail "invalid generation trace")
  fun tokenizerTests tokenizer fixture = Protocol.lines fixture (fn line =>
      let val text = Protocol.unhex line
          val ids = Tokenizer.encode tokenizer text
          val pieces = PreTokenizer.split (UnicodeNormalise.nfc text)
      in print (Protocol.ids ids ^ "\t" ^ Protocol.hex (Tokenizer.decode tokenizer ids)
                ^ "\t" ^ String.concatWith "," (List.map Protocol.hex pieces) ^ "\n") end)
  fun run (options:Options.t) =
      case #mode options of
        Options.Help => print Options.usage
      | Options.RenderTest fixture => renderTests fixture
      | Options.Chat => Util.fail "chat mode requires ChatApp"
      | mode => Host.withModel options (fn model =>
          case mode of
            Options.Encode fixture => print (Protocol.ids (Tokenizer.encode (Tokenizer.create model) (Util.readFile fixture)) ^ "\n")
          | Options.Decode ids => print (Tokenizer.decode (Tokenizer.create model) (Protocol.parseIds ids))
          | Options.TokenizerTest fixture => tokenizerTests (Tokenizer.create model) fixture
          | _ => Host.withEngine options model (fn (engine,loadSeconds) =>
              case mode of
                Options.Weights => Engine.useParams engine Inspect.weights
              | Options.Trace fixture => trace engine fixture
              | Options.Benchmark fixture => print (Json.toString
                  (Benchmark.run engine (Protocol.parseIds (Util.readFile fixture))
                    (#benchmarkCount options) (#benchmarkRuns options) loadSeconds) ^ "\n")
              | _ => Util.fail "invalid diagnostic mode"))
end
