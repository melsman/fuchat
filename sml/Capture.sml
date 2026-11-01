(* Consume control-token IDs, not substrings of decoded chunks. Reasoning and
   tool payloads stay out of normal output even across generation batches. *)
structure Capture = struct
  datatype region = Answer | Reasoning | Tool
  type t = {tokenizer:Tokenizer.t, mode:region ref, answer:string list ref,
            reason:string option ref, pending:string list ref, calls:ToolCall.t list ref,
            warnings:string list ref, invalid:bool ref, thinkStart:int, thinkEnd:int,
            toolStart:int, toolEnd:int}
  fun create tokenizer : t =
      {tokenizer=tokenizer,mode=ref Answer,answer=ref [],reason=ref NONE,pending=ref [],calls=ref [],
       warnings=ref [],invalid=ref false,thinkStart=Tokenizer.tokenId tokenizer "<think>",
       thinkEnd=Tokenizer.tokenId tokenizer "</think>",toolStart=Tokenizer.tokenId tokenizer "<tool_call>",
       toolEnd=Tokenizer.tokenId tokenizer "</tool_call>"}
  fun warn (state:t) text = #warnings state := text :: !(#warnings state)
  fun payload (state:t) = #1 (Unicode.clean true (String.concat (List.rev (!(#pending state)))))
  fun feed (state:t) id =
      let
        fun raw () = Tokenizer.raw (#tokenizer state) (Vector.fromList [id])
        fun collect () = (#pending state := raw () :: !(#pending state); "")
        fun reasonEnd () = (#reason state := SOME (Unicode.trim (payload state)); #pending state := []; #mode state := Answer; "")
        fun toolEnd () =
            let val () = if !(#invalid state) then ()
                         else ((#calls state := ToolCall.parse (Unicode.trim (payload state)) :: !(#calls state))
                               handle Fail text => warn state ("Invalid tool call: " ^ text)
                                    | Chr => warn state "Invalid UTF-8 in tool call")
            in #pending state := []; #mode state := Answer; #invalid state := false; "" end
      in
        case !(#mode state) of
          Reasoning => if id= #thinkEnd state then reasonEnd () else collect ()
        | Tool => if id= #toolEnd state then toolEnd ()
                  else if id= #toolStart state then
                    (#invalid state := true; warn state "Nested tool-call tags"; collect ()) else collect ()
        | Answer => if id= #thinkStart state then (#mode state := Reasoning; #pending state := []; "")
                    else if id= #toolStart state then (#mode state := Tool; #pending state := []; "")
                    else if id= #thinkEnd state orelse id= #toolEnd state then
                      (warn state "Unmatched closing control tag"; "")
                    else let val text = raw () in #answer state := text :: !(#answer state); text end
      end
  fun finish (state:t) complete =
      let
        val () = case !(#mode state) of
                   Answer => ()
                 | Reasoning => (#reason state := SOME (Unicode.trim (payload state)); warn state "Unclosed thinking block")
                 | Tool => warn state "Unclosed tool-call block; call discarded"
        val calls = List.rev (!(#calls state))
        val () = if not complete andalso not(List.null calls) then
                   warn state "Generation stopped before EOS; tool calls were not executed" else ()
        (* Invalid output is retained as a warning rather than executed. *)
        val accepted = if complete andalso List.null (!(#warnings state)) then calls else []
        val message : Chat.message =
            {role="assistant",content=Unicode.trim (#1 (Unicode.clean true (String.concat (List.rev (!(#answer state)))))),
             reasoning= !(#reason state),calls=accepted}
      in {message=message,warnings=List.rev (!(#warnings state))} end
end
