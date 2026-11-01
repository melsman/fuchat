structure ToolCall = struct
  type t = {name:string, arguments:Json.t}
  fun fromJson object : t =
      let val object = case Json.get object "function" of Json.Object fields => Json.Object fields | _ => object
          val name = Json.string (Json.get object "name")
          val arguments = Json.get object "arguments"
          val () = if String.size name>0 then () else Util.fail "tool call has an empty name"
          val () = case arguments of Json.Object _ => () | Json.String _ => ()
                                    | _ => Util.fail "tool arguments must be an object or JSON string"
      in {name=name,arguments=arguments} end
  fun parse text = fromJson (Json.fromString text)
  fun toJson ({name,arguments}:t) = Json.Object [("name",Json.String name),("arguments",arguments)]
end
