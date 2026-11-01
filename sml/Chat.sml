structure Chat = struct
  type message = {role:string,content:string}
  fun render (messages:message list) generation =
      let
        val lastUser = List.foldl (fn ((i,{role,...}:message),last) => if role="user" then i else last)
                          (~1) (ListPair.zip(List.tabulate(List.length messages,fn i => i),messages))
        fun leadingNewlines content =
            let fun loop i = if i<String.size content andalso String.sub(content,i)= #"\n" then loop(i+1) else i
            in String.extract(content,loop 0,NONE) end
        fun item (i,{role,content}:message) =
            if role="user" orelse role="system" then
              "<|im_start|>" ^ role ^ "\n" ^ content ^ "<|im_end|>\n"
            else if role="assistant" then
              "<|im_start|>assistant\n" ^
              (if i > lastUser andalso i=List.length messages-1 then
                 "<think>\n\n</think>\n\n" ^ leadingNewlines content else content) ^ "<|im_end|>\n"
            else Util.fail "basic chat supports system, user, and assistant roles"
        val body = String.concat (ListPair.map item (List.tabulate(List.length messages,fn i => i),messages))
      in body ^ (if generation then "<|im_start|>assistant\n<think>\n\n</think>\n\n" else "")
      end
end
