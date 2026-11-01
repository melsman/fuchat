structure Chat = struct
  type message = {role:string, content:string, reasoning:string option, calls:ToolCall.t list}
  fun simple role content : message = {role=role,content=content,reasoning=NONE,calls=[]}
  fun toJson ({role,content,reasoning,calls}:message) =
      Json.Object [("role",Json.String role),("content",Json.String content),
                   ("reasoning_content",case reasoning of NONE => Json.Null | SOME s => Json.String s),
                   ("tool_calls",Json.Array (List.map ToolCall.toJson calls))]
  fun fromJson object : message =
      {role=Json.string (Json.get object "role"), content=Json.string (Json.get object "content"),
       reasoning=case Json.get object "reasoning_content" of Json.String s => SOME s | _ => NONE,
       calls=case Json.get object "tool_calls" of Json.Array calls => List.map ToolCall.fromJson calls | _ => []}
  fun stripNewlines left right text =
      let val n = String.size text
          fun lo i = if left andalso i<n andalso String.sub(text,i)= #"\n" then lo(i+1) else i
          val start = lo 0
          fun hi i = if right andalso i>start andalso String.sub(text,i-1)= #"\n" then hi(i-1) else i
      in String.substring(text,start,hi n-start) end
  fun pieces separator text =
      let val len = String.size separator
          val n = String.size text
          fun scan (i,start,acc) =
              if i+len>n then List.rev (String.extract(text,start,NONE)::acc)
              else if String.substring(text,i,len)=separator then
                scan(i+len,i+len,String.substring(text,start,i-start)::acc)
              else scan(i+1,start,acc)
      in scan(0,0,[]) end
  fun reasoningContent ({content,reasoning,...}:message) =
      case reasoning of
        SOME reason => (content,reason)
      | NONE => case pieces "</think>" content of
                  [_] => (content,"")
                | parts =>
                    let val prefix = stripNewlines false true (hd parts)
                        val reason = List.last (pieces "<think>" prefix)
                    in (stripNewlines true false (List.last parts),stripNewlines true false reason) end
  val toolsHeader = "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n\
                    \You are provided with function signatures within <tools></tools> XML tags:\n<tools>"
  val toolsFooter = "\n</tools>\n\nFor each function call, return a json object with function name and arguments " ^
                    "within <tool_call></tool_call> XML tags:\n<tool_call>\n" ^
                    "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n"
  fun renderWith {messages:message list,generation,thinking,tools} =
      let
        val total = List.length messages
        val indexed = ListPair.zip(List.tabulate(total,fn i => i),messages)
        fun realUser ({role,content,...}:message) = role="user" andalso
            not (String.isPrefix "<tool_response>" content andalso String.isSuffix "</tool_response>" content)
        val lastUser = List.foldl (fn ((i,message),last) => if realUser message then i else last) (total-1) indexed
        val firstSystem = case messages of ({role="system",content,...}:message)::_ => SOME content | _ => NONE
        val header = if List.null tools then
                       (case firstSystem of NONE => "" | SOME s => "<|im_start|>system\n" ^ s ^ "<|im_end|>\n")
                     else "<|im_start|>system\n" ^
                       (case firstSystem of NONE => "" | SOME s => s ^ "\n\n") ^ toolsHeader ^
                       String.concat (List.map (fn tool => "\n" ^ Json.toString tool) tools) ^ toolsFooter
        val previousTool = ref false
        fun item (i,message as {role,content,calls,...}:message) =
            let val prevTool = !previousTool
                val () = previousTool := role="tool"
            in
              if role="user" orelse (role="system" andalso i>0) then
                "<|im_start|>" ^ role ^ "\n" ^ content ^ "<|im_end|>\n"
              else if role="system" then ""
              else if role="assistant" then
                let
                  val (body,reason) = reasoningContent message
                  val prefix = if i>lastUser andalso (i=total-1 orelse reason<>"") then
                                 "<think>\n" ^ stripNewlines true true reason ^ "\n</think>\n\n" ^ stripNewlines true false body
                               else body
                  fun call (j,{name,arguments}:ToolCall.t) =
                      (if j>0 orelse body<>"" then "\n" else "") ^
                      "<tool_call>\n{\"name\": " ^ Json.quote name ^ ", \"arguments\": " ^
                      (case arguments of Json.String s => s | value => Json.toString value) ^ "}\n</tool_call>"
                in "<|im_start|>assistant\n" ^ prefix ^
                   String.concat (ListPair.map call (List.tabulate(List.length calls,fn j => j),calls)) ^ "<|im_end|>\n" end
              else if role="tool" then
                (if prevTool then "" else "<|im_start|>user") ^ "\n<tool_response>\n" ^ content ^ "\n</tool_response>" ^
                (if i=total-1 orelse #role(List.nth(messages,i+1))<>"tool" then "<|im_end|>\n" else "")
              else Util.fail ("unknown chat role: " ^ role)
            end
        val body = String.concat (List.map item indexed)
      in header ^ body ^ (if generation then "<|im_start|>assistant\n" ^
                            (if thinking then "" else "<think>\n\n</think>\n\n") else "") end
  fun render messages generation = renderWith {messages=messages,generation=generation,thinking=false,tools=[]}
end
