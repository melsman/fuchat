structure Gguf = struct
  datatype value = Integer of int | Text of string | Items of value vector | Other
  type reader = {fd:Posix.IO.file_desc, buffer:Word8Vector.vector ref,
                 index:int ref, position:int ref, size:int}
  type tensor = {name:string, shape:int list, kind:int, offset:int}
  type model = {path:string, metadata:(string * value) list, tensors:tensor list, data:int}

  fun openReader path : reader =
      {fd=Posix.FileSys.openf (path,Posix.FileSys.O_RDONLY,Posix.FileSys.O.flags []),
       buffer=ref (Word8Vector.fromList []), index=ref 0, position=ref 0,
       size=Int.fromLarge (Position.toLarge (OS.FileSys.fileSize path))}
  fun close (r:reader) = Posix.IO.close (#fd r)
  fun seek (r:reader) position =
      if position < 0 orelse position > #size r then Util.fail "GGUF seek outside file"
      else
        let
          val fd = Int64.fromInt (SysWord.toInt (Posix.FileSys.fdToWord (#fd r)))
          val result = prim ("@fuchat_seek", (fd : Int64.int, Int64.fromInt position : Int64.int)) : Int64.int
          val () = if result = Int64.fromInt position then () else Util.fail "GGUF seek failed"
        in #buffer r := Word8Vector.fromList []; #index r := 0; #position r := position
        end
  fun byte (r:reader) =
      let
        val () = if !(#index r) < Word8Vector.length (!(#buffer r)) then ()
                 else (#buffer r := Posix.IO.readVec (#fd r,65536); #index r := 0)
        val () = if Word8Vector.length (!(#buffer r)) = 0 then Util.fail "truncated GGUF" else ()
        val b = Word8Vector.sub (!(#buffer r), !(#index r))
      in #index r := !(#index r) + 1; #position r := !(#position r) + 1; Word8.toInt b
      end
  fun unsigned r n =
      let fun loop (0,_,acc) = acc
            | loop (left,mul,acc) = loop (left-1,mul*256,acc + IntInf.fromInt (byte r)*mul)
      in loop (n,1,0)
      end
  fun natural r n = IntInf.toInt (unsigned r n)
  fun count (r:reader) =
      let val n = natural r 8
      in if n > #size r then Util.fail "invalid GGUF count" else n
      end
  fun text r =
      let val n = count r
      in CharVector.tabulate (n,fn _ => Char.chr (byte r))
      end
  fun signed r n =
      let val v = unsigned r n
          val modulus = IntInf.pow (256,n)
      in IntInf.toInt (if v >= modulus div 2 then v-modulus else v)
      end
  fun readValue r depth tag =
      if depth > 8 then Util.fail "GGUF metadata arrays nested too deeply"
      else case tag of
        0 => Integer (natural r 1) | 1 => Integer (signed r 1)
      | 2 => Integer (natural r 2) | 3 => Integer (signed r 2)
      | 4 => Integer (natural r 4) | 5 => Integer (signed r 4)
      | 6 => (ignore (unsigned r 4); Other)
      | 7 => Integer (natural r 1)
      | 8 => Text (text r)
      | 9 => let val kind = natural r 4
                 val n = count r
             in Items (Vector.tabulate (n,fn _ => readValue r (depth+1) kind))
             end
      | 10 => Integer (natural r 8) | 11 => Integer (signed r 8)
      | 12 => (ignore (unsigned r 8); Other)
      | _ => Util.fail ("unknown GGUF metadata type " ^ Int.toString tag)

  fun field ({metadata,...}:model) key =
      case List.find (fn (k,_) => k=key) metadata of
        SOME (_,v) => v | NONE => Util.fail ("missing GGUF metadata: " ^ key)
  fun asInt (Integer i) = i | asInt _ = Util.fail "expected GGUF integer"
  fun asText (Text s) = s | asText _ = Util.fail "expected GGUF string"
  fun asItems (Items v) = v | asItems _ = Util.fail "expected GGUF array"
  fun integer model key = asInt (field model key)
  fun string model key = asText (field model key)
  fun strings model key = Vector.map asText (asItems (field model key))
  fun integers model key = Vector.map asInt (asItems (field model key))
  fun tensor ({tensors,...}:model) name =
      case List.find (fn (t:tensor) => #name t = name) tensors of
        SOME t => t | NONE => Util.fail ("missing GGUF tensor: " ^ name)

  fun load path : model =
      let
        val r = openReader path
        fun parse () =
            let
              val magic = CharVector.tabulate (4,fn _ => Char.chr (byte r))
              val () = if magic = "GGUF" then () else Util.fail "expected little-endian GGUF"
              val version = natural r 4
              val () = if version = 2 orelse version = 3 then () else Util.fail "supported GGUF versions: 2 and 3"
              val nt = count r
              val nk = count r
              val keys = Vector.tabulate (nk,fn _ =>
                  let val key = text r
                      val tag = natural r 4
                  in (key,readValue r 0 tag)
                  end)
              val metadata = Vector.foldr (op ::) [] keys
              val alignment = case List.find (fn (k,_) => k="general.alignment") metadata of
                                NONE => 32 | SOME (_,v) => asInt v
              val () = if alignment > 0 andalso alignment <= 1048576 then () else Util.fail "invalid GGUF alignment"
              val descriptors = Vector.tabulate (nt,fn _ =>
                  let val name = text r
                      val rank = natural r 4
                      val () = if rank > 0 andalso rank <= 4 then () else Util.fail "unsupported GGUF tensor rank"
                      val dims = Vector.tabulate (rank,fn _ => count r)
                      val kind = natural r 4
                      val offset = count r
                  in {name=name,shape=Vector.foldl (op ::) [] dims,kind=kind,offset=offset}
                  end)
              val pos = !(#position r)
              val data = pos + (alignment - pos mod alignment) mod alignment
              val () = if data <= #size r then () else Util.fail "truncated GGUF tensor directory"
            in {path=path,metadata=metadata,tensors=Vector.foldr (op ::) [] descriptors,data=data}
            end
      in Util.finally parse (fn () => close r)
      end

  (* One host field at a time, read in bounded chunks. A destination permutation
     implements transpose/reshape without materializing additional tensors. *)
  val half = Real32Array.tabulate (65536,fn h =>
      Real32.fromLarge IEEEReal.TO_NEAREST
        (Real16.toLarge (PackReal16Little.fromBytes
          (Word8Vector.fromList [Word8.fromInt (h mod 256),Word8.fromInt (h div 256)]))))
  fun elements (t:tensor) = List.foldl (op *) 1 (#shape t)
  fun readTensor (model:model) (r:reader) (t:tensor) dst permute =
      let
        val width = case #kind t of 0 => 4 | 1 => 2
                    | _ => Util.fail ("quantized tensor unsupported: " ^ #name t)
        val n = elements t
        val start = #data model + #offset t
        val () = if n <= (#size r - start) div width then () else Util.fail ("truncated tensor: " ^ #name t)
        val () = seek r start
        fun exact amount =
            let
              fun read (0,parts) = Word8Vector.concat (List.rev parts)
                | read (left,parts) =
                    let val part = Posix.IO.readVec (#fd r,left)
                        val got = Word8Vector.length part
                    in if got = 0 then Util.fail "truncated tensor data"
                       else if got = left andalso List.null parts then part
                       else read (left-got,part::parts)
                    end
            in read (amount,[]) end
        fun loop done =
            if done = n then ()
            else
              let
                val count = Int.min (16384,n-done)
                val bytes = exact (count*width)
                fun value i = if width = 4 then PackReal32Little.subVec (bytes,i)
                              else Real32Array.sub (half,Word8.toInt (Word8Vector.sub(bytes,2*i))
                                                       + 256*Word8.toInt (Word8Vector.sub(bytes,2*i+1)))
                val () = Util.for count (fn i => Real32Array.update (dst,permute (done+i),value i))
              in loop (done+count)
              end
      in loop 0
      end
end
