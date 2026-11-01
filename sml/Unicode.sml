structure Unicode = struct
  fun encode cp =
      let val c = Char.chr
      in
        String.implode (if cp < 128 then [c cp]
          else if cp < 2048 then [c (192+cp div 64),c (128+cp mod 64)]
          else if cp < 65536 then [c (224+cp div 4096),c (128+cp div 64 mod 64),c (128+cp mod 64)]
          else [c (240+cp div 262144),c (128+cp div 4096 mod 64),c (128+cp div 64 mod 64),c (128+cp mod 64)])
      end
  (* Returns the scalar and the offset of its following byte. *)
  fun next s i =
      let
        val n = String.size s
        fun byte j = if j < n then Char.ord (String.sub(s,j)) else raise Chr
        fun cont j = let val b = byte j in if b >= 128 andalso b < 192 then b-128 else raise Chr end
        val b = byte i
        val (cp,len,lo) = if b < 128 then (b,1,0)
          else if b >= 194 andalso b < 224 then ((b-192)*64+cont(i+1),2,128)
          else if b >= 224 andalso b < 240 then ((b-224)*4096+cont(i+1)*64+cont(i+2),3,2048)
          else if b >= 240 andalso b < 245 then ((b-240)*262144+cont(i+1)*4096+cont(i+2)*64+cont(i+3),4,65536)
          else raise Chr
      in if cp < lo orelse cp > 1114111 orelse (cp >= 55296 andalso cp <= 57343) then raise Chr
         else (cp,i+len)
      end
  fun scalars s =
      let fun loop (i,acc) = if i=String.size s then Vector.fromList (List.rev acc)
                            else let val (cp,j) = next s i in loop(j,(cp,i)::acc) end
      in loop(0,[])
      end
  fun category cp =
      let
        val table = UnicodeCategories.ranges
        fun search (lo,hi) =
            if lo=hi then 0
            else let val mid = lo+(hi-lo) div 2
                     val (first,last,kind) = Vector.sub(table,mid)
                 in if cp < first then search(lo,mid)
                    else if cp > last then search(mid+1,hi) else kind
                 end
      in search(0,Vector.length table)
      end
  fun space cp = (cp >= 9 andalso cp <= 13) orelse cp=32 orelse cp=133 orelse cp=160
                 orelse cp=5760 orelse (cp >= 8192 andalso cp <= 8202) orelse cp=8232
                 orelse cp=8233 orelse cp=8239 orelse cp=8287 orelse cp=12288

  (* ByteLevelDecoder uses replacement for malformed/incomplete UTF-8. The
     streaming caller retains an incomplete final scalar until more arrives. *)
  fun clean final s =
      let
        val n = String.size s
        fun loop (i,parts) =
            if i=n then (String.concat (List.rev parts),"")
            else
              let
                val b = Char.ord (String.sub(s,i))
                val len = if b < 128 then 1 else if b >= 194 andalso b < 224 then 2
                          else if b >= 224 andalso b < 240 then 3
                          else if b >= 240 andalso b < 245 then 4 else 0
                fun valid j =
                    let val c = Char.ord (String.sub(s,i+j))
                    in c >= 128 andalso c < 192 andalso
                       (j <> 1 orelse ((b <> 224 orelse c >= 160) andalso (b <> 237 orelse c < 160)
                         andalso (b <> 240 orelse c >= 144) andalso (b <> 244 orelse c < 144)))
                    end
                fun prefix j = if j=len orelse i+j=n orelse not (valid j) then j else prefix(j+1)
                val consumed = if len=0 then 1 else prefix 1
              in
                if len > 0 andalso consumed=len then loop(i+len,String.substring(s,i,len)::parts)
                else if len > 0 andalso i+consumed=n andalso not final then
                  (String.concat (List.rev parts),String.extract(s,i,NONE))
                else loop(i+consumed,"\239\191\189"::parts)
              end
      in loop(0,[])
      end
  fun trim s =
      let
        val chars = scalars s
        val n = Vector.length chars
        fun white i = let val cp = #1 (Vector.sub(chars,i))
                      in space cp orelse (cp>=28 andalso cp<=31) end
        fun left i = if i<n andalso white i then left(i+1) else i
        val lo = left 0
        fun right i = if i>lo andalso white(i-1) then right(i-1) else i
        val hi = right n
        fun offset i = if i=n then String.size s else #2 (Vector.sub(chars,i))
      in String.substring(s,offset lo,offset hi-offset lo)
      end

end
