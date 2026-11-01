structure Tokenizer :> TOKENIZER = struct
  type t = {vocab:int Strings.t, bytes:string vector, byteIds:int vector,
            merges:(int * int) Pairs.t, specials:(string * int) list Array.array}
  fun byteAlphabet () =
      let
        fun direct b = (b>=33 andalso b<=126) orelse (b>=161 andalso b<=172) orelse b>=174
        val extra = ref 256
      in Vector.tabulate (256,fn b =>
           if direct b then b else let val cp = !extra in extra := cp+1; cp end)
      end
  fun create model : t =
      let
        val tokens = Gguf.strings model "tokenizer.ggml.tokens"
        val types = Gguf.integers model "tokenizer.ggml.token_type"
        val () = if Gguf.string model "tokenizer.ggml.model" = "gpt2" andalso Vector.length types=Vector.length tokens
                 then () else Util.fail "expected a byte-level BPE vocabulary"
        val vocab = Strings.create (Vector.length tokens*2+1)
        val () = Vector.appi (fn (i,s) => Strings.insert vocab (s,i)) tokens
        fun id token = case Strings.find vocab token of SOME i => i
                       | NONE => Util.fail ("token absent from vocabulary: " ^ token)
        val alphabet = byteAlphabet ()
        val inverse = Array.array (512,~1)
        val () = Vector.appi (fn (b,cp) => Array.update(inverse,cp,b)) alphabet
        val specials = Array.array (256,[])
        val () = Vector.appi (fn (i,token) =>
            if Vector.sub(types,i)=3 orelse Vector.sub(types,i)=4 then
              if String.size token=0 then Util.fail "empty special token"
              else let val b = Char.ord(String.sub(token,0))
                   in Array.update(specials,b,(token,i)::Array.sub(specials,b)) end
            else ()) tokens
        fun raw (i,token) =
            if Vector.sub(types,i)=3 orelse Vector.sub(types,i)=4 then token
            else String.implode (Vector.foldr (fn ((cp,_),acc) =>
                   if cp < Array.length inverse andalso Array.sub(inverse,cp) >= 0 then
                     Char.chr(Array.sub(inverse,cp))::acc
                   else Util.fail "vocabulary contains a non-byte BPE symbol") [] (Unicode.scalars token))
        val bytes = Vector.mapi raw tokens
        val byteIds = Vector.map (id o Unicode.encode) alphabet
        val rules = Gguf.strings model "tokenizer.ggml.merges"
        val merges = Pairs.create (Vector.length rules*2+1)
        val () = Vector.appi (fn (rank,rule) =>
            case String.fields (fn c => c = #" ") rule of
              [a,b] => Pairs.insert merges ((id a,id b),(rank,id (a^b)))
            | _ => Util.fail ("invalid BPE merge: " ^ rule)) rules
      in {vocab=vocab,bytes=bytes,byteIds=byteIds,merges=merges,specials=specials}
      end
  fun tokenId (t:t) name =
      case Strings.find (#vocab t) name of SOME i => i | NONE => Util.fail ("unknown token: " ^ name)

  (* A linked sequence and rank heap avoid rescanning long pieces after every
     merge. Stale heap entries are checked against their live neighbors. *)
  fun bpe (t:t) piece =
      let
        val n = String.size piece
        val ids = Array.tabulate (n,fn i => Vector.sub(#byteIds t,Char.ord(String.sub(piece,i))))
        val next = Array.tabulate (n,fn i => if i+1=n then ~1 else i+1)
        val prev = Array.tabulate (n,fn i => i-1)
        val alive = Array.array (n,true)
        type candidate = int * int * int * int * int
        val heap = Array.array (3*n+1,NONE : candidate option)
        val used = ref 0
        fun entry i = valOf(Array.sub(heap,i))
        fun less ((rank,left,_,_,_):candidate,(r,l,_,_,_):candidate) = rank < r orelse (rank=r andalso left<l)
        fun push value =
            let
              val i = !used
              val () = used := i+1
              fun up j = if j=0 then Array.update(heap,0,SOME value)
                         else let val p = (j-1) div 2
                              in if less(value,entry p) then
                                   (Array.update(heap,j,Array.sub(heap,p)); up p)
                                 else Array.update(heap,j,SOME value)
                              end
            in up i
            end
        fun pop () =
            if !used=0 then NONE
            else
              let
                val result = entry 0
                val count = !used-1
                val last = entry count
                val () = used := count
                val () = Array.update(heap,count,NONE)
                fun down i =
                    if 2*i+1 >= count then Array.update(heap,i,SOME last)
                    else let val l = 2*i+1
                             val r = l+1
                             val child = if r<count andalso less(entry r,entry l) then r else l
                         in if less(entry child,last) then
                              (Array.update(heap,i,Array.sub(heap,child)); down child)
                            else Array.update(heap,i,SOME last)
                         end
                val () = if count>0 then down 0 else ()
              in SOME result
              end
        fun offer left =
            if left<0 then ()
            else let val right = Array.sub(next,left)
                 in if right<0 then ()
                    else case Pairs.find (#merges t) (Array.sub(ids,left),Array.sub(ids,right)) of
                           NONE => ()
                         | SOME (rank,_) => push(rank,left,right,Array.sub(ids,left),Array.sub(ids,right))
                 end
        val () = Util.for n offer
        fun merge () =
            case pop () of
              NONE => ()
            | SOME (_,left,right,a,b) =>
              if not (Array.sub(alive,left) andalso Array.sub(alive,right))
                 orelse Array.sub(next,left) <> right
                 orelse Array.sub(ids,left) <> a orelse Array.sub(ids,right) <> b then merge ()
              else
                let
                  val (_,combined) = valOf(Pairs.find (#merges t) (a,b))
                  val after = Array.sub(next,right)
                  val () = Array.update(ids,left,combined)
                  val () = Array.update(next,left,after)
                  val () = Array.update(alive,right,false)
                  val () = if after<0 then () else Array.update(prev,after,left)
                  val () = offer (Array.sub(prev,left))
                  val () = offer left
                in merge ()
                end
        val () = merge ()
        fun collect (i,acc) = if i<0 then List.rev acc else collect(Array.sub(next,i),Array.sub(ids,i)::acc)
      in if n=0 then [] else collect(0,[])
      end

  fun encode (t:t) input =
      let
        val n = String.size input
        fun special i =
            let
              fun matches token =
                  let val len = String.size token
                  in i+len<=n andalso String.substring(input,i,len)=token end
              fun choose ((token,id),best) =
                  if matches token then
                    (case best of NONE => SOME(token,id)
                     | SOME (old,_) => if String.size token > String.size old then SOME(token,id) else best)
                  else best
            in List.foldl choose NONE (Array.sub(#specials t,Char.ord(String.sub(input,i))))
            end
        fun normal text acc =
            List.foldl (fn (piece,out) => List.revAppend(bpe t piece,out)) acc
              (PreTokenizer.split (UnicodeNormalise.nfc text))
        fun loop (i,start,acc) =
            if i=n then Vector.fromList (List.rev (normal (String.substring(input,start,n-start)) acc))
            else case special i of
              NONE => loop(i+1,start,acc)
            | SOME (token,id) =>
              let val acc = normal (String.substring(input,start,i-start)) acc
                  val j = i+String.size token
              in loop(j,j,id::acc)
              end
      in loop(0,0,[])
      end
  fun raw (t:t) ids = String.concat (Vector.foldr (fn (id,parts) =>
      if id>=0 andalso id<Vector.length(#bytes t) then Vector.sub(#bytes t,id)::parts
      else Util.fail "invalid token ID during decoding") [] ids)
  fun decode t ids = #1 (Unicode.clean true (raw t ids))
end
