structure Protocol = struct
  fun ids values = String.concatWith "," (Vector.foldr (fn (i,acc) => Int.toString i::acc) [] values)
  fun parseIds s = Vector.fromList (List.map Util.int (String.tokens (fn c => c = #"," orelse Char.isSpace c) s))
  fun hex s =
      let val digits = "0123456789abcdef"
      in String.concat (List.map (fn c => let val n = Char.ord c
                                         in String.implode [String.sub(digits,n div 16),String.sub(digits,n mod 16)] end)
                                  (String.explode s))
      end
  fun unhex s =
      let
        fun digit c = if c>= #"0" andalso c<= #"9" then Char.ord c-48
                      else if c>= #"a" andalso c<= #"f" then Char.ord c-87
                      else if c>= #"A" andalso c<= #"F" then Char.ord c-55 else Util.fail "invalid hex input"
        val () = if String.size s mod 2=0 then () else Util.fail "odd hex length"
      in CharVector.tabulate (String.size s div 2,fn i => Char.chr (16*digit(String.sub(s,2*i))+digit(String.sub(s,2*i+1))))
      end
  fun lines path f =
      let val stream = TextIO.openIn path
          fun loop () = case TextIO.inputLine stream of NONE => () | SOME s => (f (Util.stripNewline s); loop ())
      in Util.finally loop (fn () => TextIO.closeIn stream)
      end
end
