structure Util = struct
  fun fail s = raise Fail s
  fun finally f cleanup =
      let val result = f () handle ex => (cleanup (); raise ex)
      in cleanup (); result
      end
  fun for n f =
      let fun loop i = if i = n then () else (f i; loop (i + 1))
      in loop 0
      end
  fun int s = case Int.fromString s of SOME n => n | NONE => fail ("invalid integer: " ^ s)
  fun log s = (TextIO.output (TextIO.stdErr, s ^ "\n"); TextIO.flushOut TextIO.stdErr)
  fun readFile path =
      let val stream = TextIO.openIn path
      in finally (fn () => TextIO.inputAll stream) (fn () => TextIO.closeIn stream)
      end
  fun stripNewline s =
      let fun stop i = if i > 0 andalso (String.sub (s, i-1) = #"\n" orelse String.sub (s, i-1) = #"\r")
                       then stop (i-1) else i
      in String.substring (s, 0, stop (String.size s))
      end
  fun trim s =
      let val n = String.size s
          fun left i = if i < n andalso Char.isSpace (String.sub (s,i)) then left (i+1) else i
          val lo = left 0
          fun right i = if i > lo andalso Char.isSpace (String.sub (s,i-1)) then right (i-1) else i
      in String.substring (s, lo, right n - lo)
      end
end

(* Small mutable tables; the vocabulary and merge tables are built once. *)
functor HashTable (Key : sig type t val hash : t -> word val equal : t * t -> bool end) = struct
  type 'a t = (Key.t * 'a) list Array.array
  fun create n = Array.array (Int.max (n, 17), [])
  fun bucket table key = Word.toInt (Word.mod (Key.hash key, Word.fromInt (Array.length table)))
  fun find table key =
      let fun scan [] = NONE
            | scan ((k,v)::rest) = if Key.equal (k,key) then SOME v else scan rest
      in scan (Array.sub (table, bucket table key))
      end
  fun insert table (key,value) =
      let val i = bucket table key
      in Array.update (table, i, (key,value) :: Array.sub (table,i))
      end
end
structure Strings = HashTable (struct
  type t = string
  val equal = op =
  fun hash s = CharVector.foldl (fn (c,h) => Word.xorb (Word.* (h,0w16777619), Word.fromInt (Char.ord c))) 0w2166136261 s
end)
structure Pairs = HashTable (struct
  type t = int * int
  val equal = op =
  fun hash (a,b) = Word.xorb (Word.* (Word.fromInt a,0w16777619),Word.fromInt b)
end)
