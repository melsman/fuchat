structure PreTokenizer = struct
  (* Left-to-right alternatives of the Qwen regex, including greedy whitespace
     backtracking. Numeric categories match one scalar, not a run of digits. *)
  fun split s =
      let
        val chars = Unicode.scalars s
        val n = Vector.length chars
        fun cp i = #1 (Vector.sub(chars,i))
        fun offset i = if i=n then String.size s else #2 (Vector.sub(chars,i))
        fun letter i = i < n andalso Unicode.category (cp i)=1
        fun number i = i < n andalso Unicode.category (cp i)=2
        fun ws i = i < n andalso Unicode.space (cp i)
        fun newline i = i < n andalso (cp i=10 orelse cp i=13)
        fun punct i = i < n andalso not (ws i orelse letter i orelse number i)
        fun run pred i = if i<n andalso pred i then run pred (i+1) else i
        fun contraction i =
            if cp i <> 39 then NONE
            else
              let
                val suffixes = ["s","t","re","ve","m","ll","d"]
                fun same c value = value=Char.ord c orelse value=Char.ord(Char.toUpper c)
                                   orelse (c= #"s" andalso value=383)
                fun matches t =
                    i+String.size t < n andalso
                    List.all (fn j => same (String.sub(t,j)) (cp(i+j+1)))
                             (List.tabulate(String.size t,fn j => j))
              in case List.find matches suffixes of NONE => NONE | SOME t => SOME(i+1+String.size t)
              end
        fun endAt i =
            case contraction i of
              SOME j => j
            | NONE =>
              if letter i then run letter (i+1)
              else if not (newline i orelse letter i orelse number i) andalso letter (i+1)
                   then run letter (i+2)
              else if number i then i+1
              else if punct i orelse (cp i=32 andalso punct (i+1)) then
                run newline (run punct (if punct i then i else i+1))
              else if ws i then
                let
                  val stop = run ws (i+1)
                  fun lastNewline j = if j < i then NONE
                                      else if newline j then SOME (j+1) else lastNewline(j-1)
                in case lastNewline (stop-1) of
                     SOME j => j
                   | NONE => if stop=n then stop else if stop-i > 1 then stop-1 else stop
                end
              else Util.fail "pre-tokenizer made no progress"
        fun loop (i,acc) = if i=n then List.rev acc
                          else let val j = endAt i
                               in loop(j,String.substring(s,offset i,offset j-offset i)::acc) end
      in loop(0,[])
      end
end
