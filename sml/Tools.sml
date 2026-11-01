structure Tools = struct
  fun description ({name,description,...}:ToolEntries.entry) =
      Json.Object [("type",Json.String "function"),("function",Json.Object
        [("name",Json.String name),("description",Json.String description),
         ("parameters",Json.Object [("type",Json.String "object"),("properties",Json.Object
           [("n",Json.Object [("type",Json.String "number"),("description",Json.String "")])])])])]
  val descriptions = List.map description ToolEntries.all
  fun decimal n = String.translate (fn #"~" => "-" | c => String.str c) (Int64.toString n)
  fun invoke ctx ({name,arguments}:ToolCall.t) =
      let
        val entry = case List.find (fn (entry:ToolEntries.entry) => #name entry=name) ToolEntries.all of
                      SOME entry => entry | NONE => Util.fail ("unknown tool: " ^ name)
        val arguments = case arguments of Json.String s => Json.fromString s | value => value
        val number = case Json.get arguments "n" of Json.Number n => n
                      | _ => Util.fail "tool argument n must be an integer"
        val n = case Int64.scan StringCvt.DEC Substring.getc (Substring.full number) of
                  SOME (n,rest) => if Substring.isEmpty rest then n else Util.fail "tool argument n must be an integer"
                | NONE => Util.fail "tool argument n is outside the i64 range"
        (* These tools materialize iota(n+1); keep model-selected allocations
           and integer sums within a predictable range. *)
        val () = if n>=0 andalso n<=1000000 then () else Util.fail "tool argument n must be between 0 and 1000000"
      in decimal (#invoke entry ctx n) end
  fun error text = Json.toString (Json.Object [("error",Json.String text)])
  fun run enabled ctx call =
      if not enabled then error "tools are disabled"
      else invoke ctx call handle Fail text => error text
                               | Overflow => error "tool argument n is outside the i64 range"
                               | qwen.Error text => error ("Futhark tool failed: " ^ text)
end
