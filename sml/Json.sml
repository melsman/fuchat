(* JSON at the model boundary: bounded nesting, strict syntax, decoded strings.
   Serialization matches Jinja's sorted, ASCII, HTML-safe tojson filter. *)
structure Json = struct
  datatype t = Object of (string * t) list | Array of t list | String of string
             | Number of string | Bool of bool | Null
  fun get (Object fields) name =
        (case List.find (fn (key,_) => key=name) fields of SOME (_,v) => v | NONE => Null)
    | get _ _ = Null
  fun string (String s) = s | string _ = Util.fail "expected JSON string"
  fun array (Array xs) = xs | array _ = Util.fail "expected JSON array"
  fun boolean (Bool b) = b | boolean _ = Util.fail "expected JSON boolean"
  fun fromString input =
      let
        val n = String.size input
        val pos = ref 0
        fun fail () = Util.fail ("invalid JSON at byte " ^ Int.toString (!pos))
        fun peek () = if !pos=n then NONE else SOME (String.sub(input,!pos))
        fun take () = case peek () of NONE => fail () | SOME c => (pos:= !pos+1; c)
        fun ws () = case peek () of SOME c => if c= #" " orelse c= #"\n" orelse c= #"\r" orelse c= #"\t"
                                              then (pos:= !pos+1; ws ()) else () | NONE => ()
        fun expect c = if take ()=c then () else fail ()
        fun hex () =
            let fun digit c = if c>= #"0" andalso c<= #"9" then Char.ord c-48
                              else if c>= #"a" andalso c<= #"f" then Char.ord c-87
                              else if c>= #"A" andalso c<= #"F" then Char.ord c-55 else fail ()
                fun loop (0,v) = v | loop (i,v) = loop(i-1,v*16+digit(take ()))
            in loop(4,0) end
        fun quoted () =
            let
              val () = expect #"\""
              fun escape () = case take () of
                  #"\"" => "\"" | #"\\" => "\\" | #"/" => "/" | #"b" => "\008" | #"f" => "\012"
                | #"n" => "\n" | #"r" => "\r" | #"t" => "\t"
                | #"u" => let val cp = hex ()
                          in if cp>=55296 andalso cp<=56319 then
                               let val () = expect #"\\"
                                   val () = expect #"u"
                                   val lo = hex ()
                               in if lo>=56320 andalso lo<=57343 then Unicode.encode (65536+(cp-55296)*1024+lo-56320)
                                  else fail () end
                             else if cp>=56320 andalso cp<=57343 then fail () else Unicode.encode cp
                          end
                | _ => fail ()
              fun loop parts =
                  case peek () of
                    NONE => fail ()
                  | SOME #"\"" => (pos:= !pos+1; String.concat (List.rev parts))
                  | SOME #"\\" => (pos:= !pos+1; loop(escape ()::parts))
                  | SOME c => if Char.ord c<32 then fail ()
                              else let val start = !pos
                                       val (_,stop) = Unicode.next input start
                                   in pos:=stop; loop(String.substring(input,start,stop-start)::parts) end
            in loop [] end
        fun number () =
            let
              val start = !pos
              fun digit () = case peek () of SOME c => Char.isDigit c | NONE => false
              fun digits () = if digit () then (pos:= !pos+1; digits ()) else ()
              val () = if peek ()=SOME #"-" then pos:= !pos+1 else ()
              val () = if peek ()=SOME #"0" then pos:= !pos+1
                       else if digit () then digits () else fail ()
              val () = if peek ()=SOME #"." then (pos:= !pos+1; if digit () then digits () else fail ()) else ()
              val () = if peek ()=SOME #"e" orelse peek ()=SOME #"E" then
                         (pos:= !pos+1;
                          if peek ()=SOME #"+" orelse peek ()=SOME #"-" then pos:= !pos+1 else ();
                          if digit () then digits () else fail ()) else ()
            in Number (String.substring(input,start,!pos-start)) end
        fun literal text value = (List.app expect (String.explode text); value)
        fun value depth =
            if depth>64 then Util.fail "JSON nesting exceeds 64 levels"
            else (ws (); case peek () of
              SOME #"\"" => String (quoted ())
            | SOME #"{" =>
              let
                val () = pos:= !pos+1
                val () = ws ()
                fun fields acc =
                    let val key = quoted ()
                        val () = if List.exists (fn (k,_) => k=key) acc then fail () else ()
                        val () = ws ()
                        val () = expect #":"
                        val v = value (depth+1)
                        val () = ws ()
                    in case take () of #"}" => Object (List.rev ((key,v)::acc))
                                       | #"," => (ws (); fields ((key,v)::acc)) | _ => fail () end
              in if peek ()=SOME #"}" then (pos:= !pos+1; Object []) else fields [] end
            | SOME #"[" =>
              let val () = pos:= !pos+1
                  val () = ws ()
                  fun items acc =
                      let val v = value (depth+1)
                          val () = ws ()
                      in case take () of #"]" => Array (List.rev(v::acc))
                                         | #"," => items (v::acc) | _ => fail () end
              in if peek ()=SOME #"]" then (pos:= !pos+1; Array []) else items [] end
            | SOME #"t" => literal "true" (Bool true)
            | SOME #"f" => literal "false" (Bool false)
            | SOME #"n" => literal "null" Null
            | SOME c => if c= #"-" orelse Char.isDigit c then number () else fail ()
            | NONE => fail ())
        val result = value 0
        val () = ws ()
      in if !pos=n then result else fail () end

  fun quote text =
      let
        val digits = "0123456789abcdef"
        fun hex cp = "\\u" ^ String.implode (List.map (fn shift => String.sub(digits,cp div shift mod 16)) [4096,256,16,1])
        fun cp value = case value of
            34 => "\\\"" | 92 => "\\\\" | 8 => "\\b" | 12 => "\\f" | 10 => "\\n" | 13 => "\\r" | 9 => "\\t"
          | _ => if value<32 orelse value>=127 orelse value=38 orelse value=39 orelse value=60 orelse value=62 then
                   if value<65536 then hex value
                   else let val v = value-65536 in hex (55296+v div 1024) ^ hex(56320+v mod 1024) end
                 else String.str (Char.chr value)
      in "\"" ^ String.concat (Vector.foldr (fn ((value,_),acc) => cp value::acc) [] (Unicode.scalars text)) ^ "\"" end
  fun sort fields =
      let fun insert (entry as (key,_),[]) = [entry]
            | insert (entry as (key,_),(head as (k,_))::rest) =
                if String.compare(key,k)=LESS then entry::head::rest else head::insert(entry,rest)
      in List.foldl insert [] fields end
  fun toString value = case value of
      Object fields => "{" ^ String.concatWith ", " (List.map (fn (key,v) => quote key ^ ": " ^ toString v) (sort fields)) ^ "}"
    | Array values => "[" ^ String.concatWith ", " (List.map toString values) ^ "]"
    | String s => quote s | Number n => n | Bool true => "true" | Bool false => "false" | Null => "null"
end
