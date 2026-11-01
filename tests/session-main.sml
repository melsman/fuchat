(* Replay scripted model token batches through the production Session module,
   while executing real typed Futhark tools in a weight-free context. *)
val model = Gguf.load (hd (CommandLine.arguments ()))
val tokenizer = Tokenizer.create model
val eos = Gguf.integer model "tokenizer.ggml.eos_token_id"
fun number object key default = case Json.get object key of Json.Number n => Util.int n | _ => default
fun flag object key default = case Json.get object key of Json.Bool b => b | _ => default
fun process test =
    let
      val ctx = qwen.Context.new qwen.Config.default
      val cached = ref (Vector.fromList [] : int vector)
      val prompts = ref ([]:Json.t list)
      val executions = ref ([]:Json.t list)
      val output = ref ([]:string list)
      val warnings = ref ([]:string list)
      val messages = ref (case Json.get test "messages" of Json.Array ms => List.map Chat.fromJson ms | _ => [])
      fun execute enabled call =
          let val result = Tools.run enabled ctx call
          in executions:=Json.Object [("call",ToolCall.toJson call),("result",Json.String result)]:: !executions; result end
      fun turn input =
          let
            val options = {thinking=flag input "thinking" false,tools=flag input "tools" false,show=flag input "show" false,
                           count=number input "count" 5,limit=number input "limit" 4096,
                           context=number input "context" 8192,maxToolRounds=number input "rounds" 8}
            val responses = ref (List.map (fn value => Vector.concat [Tokenizer.encode tokenizer (Json.string value),Vector.fromList [eos]])
                                  (Json.array (Json.get input "responses")))
            val offset = ref 0
            fun generate ids amount =
                let
                  val response = case !responses of x::_ => x | [] => Util.fail "scripted model ran out of responses"
                  val reused = Int.min (Engine.prefix (!cached) ids,Vector.length ids-1)
                  val first = !offset=0
                  val () = if first then prompts:=Json.Object [("text",Json.String (Tokenizer.decode tokenizer ids)),
                            ("ids",Json.Array (Vector.foldr (fn (i,xs) => Json.Number(Int.toString i)::xs) [] ids)),
                            ("reused",Json.Number (Int.toString reused))]:: !prompts else ()
                  val count = Int.min (amount,Vector.length response- !offset)
                  val new = Vector.tabulate (count,fn i => Vector.sub(response,i+ !offset))
                  val () = offset:= !offset+count
                  val () = if !offset=Vector.length response then (responses:=tl (!responses); offset:=0) else ()
                  val all = Vector.concat [ids,new]
                  val () = cached:=Vector.tabulate(Vector.length all-1,fn i => Vector.sub(all,i))
                in {ids=all,newTokens=new,reused=reused,eos=eos} end
            val text = Json.string (Json.get input "input")
            val proposed = !messages @ [Chat.simple "user" text]
            val result = Session.reply tokenizer options generate (execute (#tools options))
                         (fn text => output:=text:: !output) (fn text => warnings:=text:: !warnings) proposed
          in messages:= #messages result end
      fun run () =
          (List.app turn (Json.array (Json.get test "turns"));
           Json.Object [("messages",Json.Array (List.map Chat.toJson (!messages))),
             ("visible",Json.String (String.concat (List.rev (!output)))),
             ("warnings",Json.Array (List.map Json.String (List.rev (!warnings)))),
             ("prompts",Json.Array (List.rev (!prompts))),
             ("executions",Json.Array (List.rev (!executions)))])
    in Util.finally run (fn () => qwen.Context.free ctx) end
val () = Protocol.lines (List.nth(CommandLine.arguments (),1))
    (fn line => print (Json.toString (process (Json.fromString line)) ^ "\n"))
