structure Gguf :> GGUF = struct
  datatype value = Integer of int | Text of string | Items of value vector | Other
  type reader = {fd:Posix.IO.file_desc, buffer:Word8Array.array,
                 index:int ref, available:int ref, position:int ref, size:int}
  type tensor = {name:string, shape:int list, kind:int, offset:int}
  type model = {path:string, metadata:(string * value) vector, tensors:tensor vector, data:int}

  fun openReader path : reader =
      let val size = Int.fromLarge (Position.toLarge (OS.FileSys.fileSize path))
          val buffer = Word8Array.array(65536,0w0)
          val index = ref 0
          val available = ref 0
          val position = ref 0
          val fd = Posix.FileSys.openf (path,Posix.FileSys.O_RDONLY,Posix.FileSys.O.flags [])
      in {fd=fd,buffer=buffer,index=index,available=available,position=position,size=size} end
  fun close (r:reader) = Posix.IO.close (#fd r)
  fun withReader (model:model) action =
      let val reader = openReader (#path model)
      in Util.finally (fn () => action reader) (fn () => close reader) end
  fun seek (r:reader) position =
      if position < 0 orelse position > #size r then Util.fail "GGUF seek outside file"
      else
        let
          val fd = Int64.fromInt (SysWord.toInt (Posix.FileSys.fdToWord (#fd r)))
          val result = prim ("@fuchat_seek", (fd : Int64.int, Int64.fromInt position : Int64.int)) : Int64.int
          val () = if result = Int64.fromInt position then () else Util.fail "GGUF seek failed"
        in #available r := 0; #index r := 0; #position r := position
        end
  fun refill (r:reader) =
      (#available r := Posix.IO.readArr (#fd r,Word8ArraySlice.full (#buffer r));
       #index r := 0;
       if !(#available r)=0 then Util.fail "truncated GGUF" else ())
  fun byte (r:reader) =
      let val () = if !(#index r)< !(#available r) then () else refill r
          val b = Word8Array.sub (#buffer r,!(#index r))
      in #index r := !(#index r)+1; #position r := !(#position r)+1; Word8.toInt b end
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
  fun text (r:reader) =
      let val n = count r
          val () = if n<= #size r- !(#position r) then () else Util.fail "truncated GGUF string"
          val out = Word8Array.array(n,0w0)
          fun copy done =
              if done=n then ()
              else let val () = if !(#index r)< !(#available r) then () else refill r
                       val amount = Int.min(n-done,!(#available r)- !(#index r))
                       val () = Word8ArraySlice.copy
                           {src=Word8ArraySlice.slice(#buffer r,!(#index r),SOME amount),dst=out,di=done}
                       val () = #index r := !(#index r)+amount
                       val () = #position r := !(#position r)+amount
                   in copy(done+amount) end
          val () = copy 0
      in Byte.bytesToString (Word8Array.vector out) end
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
      case Vector.find (fn (k,_) => k=key) metadata of
        SOME (_,v) => v | NONE => Util.fail ("missing GGUF metadata: " ^ key)
  fun asInt (Integer i) = i | asInt _ = Util.fail "expected GGUF integer"
  fun asText (Text s) = s | asText _ = Util.fail "expected GGUF string"
  fun asItems (Items v) = v | asItems _ = Util.fail "expected GGUF array"
  fun integer model key = asInt (field model key)
  fun string model key = asText (field model key)
  fun strings model key = Vector.map asText (asItems (field model key))
  fun integers model key = Vector.map asInt (asItems (field model key))
  fun tensor ({tensors,...}:model) name =
      case Vector.find (fn (t:tensor) => #name t = name) tensors of
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
              val metadata = keys
              val alignment = case Vector.find (fn (k,_) => k="general.alignment") metadata of
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
            in {path=path,metadata=metadata,tensors=descriptors,data=data}
            end
      in Util.finally parse (fn () => close r)
      end

  (* A borrowed reusable byte buffer; no fragment lists or readVec allocations.
     Chunk boundaries are whole rows, so consumers can write contiguous slices. *)
  val half = Real32Array.tabulate (65536,fn h =>
      Real32.fromLarge IEEEReal.TO_NEAREST
        (Real16.toLarge (PackReal16Little.fromBytes
          (Word8Vector.tabulate(2,fn i => Word8.fromInt (if i=0 then h mod 256 else h div 256))))))
  fun elements (t:tensor) = List.foldl (op *) 1 (#shape t)
  fun readRows unpack32 unpack16 (model:model) (r:reader) (t:tensor) columns consume =
      let
        val width = case #kind t of 0 => 4 | 1 => 2
                    | _ => Util.fail ("quantized tensor unsupported: " ^ #name t)
        val n = elements t
        val () = if columns>0 andalso n mod columns=0 then () else Util.fail "invalid tensor row width"
        val start = #data model + #offset t
        val () = if start >= #data model andalso start <= #size r andalso n <= (#size r-start) div width
                 then () else Util.fail ("truncated tensor: " ^ #name t)
        val () = seek r start
        val rowBytes = columns*width
        val buffer = if rowBytes<=Word8Array.length (#buffer r) then #buffer r
                     else Word8Array.array(rowBytes,0w0)
        val batch = Word8Array.length buffer div rowBytes
        val rows = n div columns
        val unpack = if width=4 then unpack32 else unpack16
        fun exact amount =
            let fun fill done =
                    if done=amount then ()
                    else let val got = Posix.IO.readArr (#fd r,Word8ArraySlice.slice(buffer,done,SOME(amount-done)))
                         in if got=0 then Util.fail "truncated tensor data" else fill(done+got) end
            in fill 0; #position r := !(#position r)+amount end
        fun loop first =
            if first=rows then ()
            else let val count = Int.min(batch,rows-first)
                     val () = exact (count*rowBytes)
                     val () = consume (first,count,fn i => unpack(buffer,i))
                 in loop(first+count) end
      in loop 0 end
  fun half32 (bytes,i) = Real32Array.sub (half,
      Word8.toInt (Word8Array.sub(bytes,2*i)) + 256*Word8.toInt (Word8Array.sub(bytes,2*i+1)))
  fun readTensor32Rows model reader tensor columns consume =
      readRows PackReal32Little.subArr half32 model reader tensor columns consume
  fun readTensor16Rows model reader tensor columns consume = readRows
      (fn bytes => Real16.fromLarge IEEEReal.TO_NEAREST (Real32.toLarge (PackReal32Little.subArr bytes)))
      PackReal16Little.subArr model reader tensor columns consume
end
