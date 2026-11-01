(* Failure paths and representation boundaries, using real GGUF/Futhark. *)
fun check condition text = if condition then () else raise Fail text
fun fails action = (action (); false) handle Fail _ => true
exception ActionFailure
exception CleanupFailure
val released = ref ([]:int list)
fun remember n () = released:=n:: !released
val scope = Resource.create ()
val () = Resource.defer scope (remember 1)
val () = Resource.defer scope (fn () => (remember 2 (); raise CleanupFailure))
val () = Resource.defer scope (remember 3)
val () = ((Resource.release scope; raise Fail "missing cleanup failure") handle CleanupFailure => ())
val () = Resource.release scope
val () = check (!released=[1,2,3]) "cleanup did not release every resource once in LIFO order"
val () = ((Resource.scope (fn r =>
             (Resource.defer r (fn () => raise CleanupFailure); raise ActionFailure));
           raise Fail "missing action failure") handle ActionFailure => ())
val () = ((Util.finally (fn () => raise ActionFailure) (fn () => raise CleanupFailure);
           raise Fail "finally masked action failure") handle ActionFailure => ())
val defaults = Options.parse 1024 []
val () = check (#context defaults=1024 andalso #count defaults=5 andalso not (#thinking defaults)) "option defaults"
val options = Options.parse 8192 ["--cs","64","--think","--tools","--benchmark","ids","--benchmark-runs","2"]
val () = check (#context options=64 andalso #thinking options andalso #tools options andalso
                #benchmarkRuns options=2 andalso #mode options=Options.Benchmark "ids") "option parsing"
val () = List.app (fn args => check (fails (fn () => ignore (Options.parse 1024 args))) "accepted invalid options")
    [["--cs","0"],["--cnt","-1"],["--cs","64junk"],["--benchmark-runs","0"],["--model"],["--unknown"]]
val args = CommandLine.arguments ()
val model = Gguf.load (List.nth(args,0))
val broken = Gguf.load (List.nth(args,1))
val edge = Gguf.load (List.nth(args,2))
val expected = Byte.stringToBytes (Util.readFile (List.nth(args,3)))
val () = Gguf.withReader edge (fn reader =>
    let val tensor = Gguf.tensor edge "half_bits"
        fun checkHalf columns =
            let val seen = ref 0
                val () = Gguf.readTensor16Rows edge reader tensor columns (fn (first,count,get) =>
                    (check (first*columns= !seen) "rows arrived out of order";
                     Util.for (count*columns) (fn i =>
                       let val bits = PackReal16Little.toBytes (get i)
                           val value = Word8.toInt (Word8Vector.sub(bits,0)) + 256*Word8.toInt (Word8Vector.sub(bits,1))
                       in check (value=first*columns+i) "f16 source bits changed" end);
                     seen:=(first+count)*columns))
            in check (!seen=65536) "incomplete buffered read" end
        val () = checkHalf 1
        val () = checkHalf 65536 (* A single row wider than the regular buffer. *)
        val narrow = Gguf.tensor edge "float_values"
        val () = Gguf.readTensor16Rows edge reader narrow 1 (fn (first,count,get) =>
            Util.for count (fn i =>
              check (PackReal16Little.toBytes (get i) = Word8Vector.tabulate(2,fn j => Word8Vector.sub(expected,2*(first+i)+j)))
                    "f32 to f16 rounding mismatch"))
    in () end)
(* Compare every cell of row-major, transpose, and attention permutations,
   including batches crossing head boundaries and protected destination ends. *)
fun layoutTest layout rows columns batch =
    let val size = rows*columns
        val out = Array.array(size+14,~1)
        fun store (start,count,get) =
            ArraySlice.modifyi (fn (i,_) => get i) (ArraySlice.slice(out,start,SOME count))
        fun chunks first =
            if first=rows then ()
            else let val count = Int.min(batch,rows-first)
                     val () = TensorLayout.fill layout {rows=rows,columns=columns,first=first,count=count,offset=7}
                         (fn i => (check (i>=0 andalso i<count*columns) "layout read outside its borrowed batch"; first*columns+i)) store
                 in chunks(first+count) end
        val () = chunks 0
        fun expected i = case layout of
            TensorLayout.RowMajor => i
          | TensorLayout.Transpose => (i mod rows)*columns+i div rows
          | TensorLayout.Attention dim =>
              ((i div (columns*dim))*dim+i mod dim)*columns+(i div dim mod columns)
        val () = ArraySlice.appi (fn (i,value) => check (value=expected i) "bulk tensor layout mismatch")
            (ArraySlice.slice(out,7,SOME size))
    in check (ArraySlice.all (fn x => x= ~1) (ArraySlice.slice(out,0,SOME 7)) andalso
              ArraySlice.all (fn x => x= ~1) (ArraySlice.slice(out,size+7,SOME 7))) "slice overwrote another field" end
val () = List.app (fn batch =>
    (layoutTest TensorLayout.RowMajor 257 17 batch;
     layoutTest TensorLayout.Transpose 257 17 batch;
     layoutTest (TensorLayout.Attention 128) 384 17 batch)) [1,31,63,128,129,511]
(* Loading a truncated model after some fields have been allocated must unwind;
   the same process must then be able to open and run a valid model. *)
val () = Util.for 3 (fn _ => check (fails (fn () => Engine.withModel broken 64 (fn _ => ()))) "accepted truncated weights")
val escaped = ref (NONE:Engine.t option)
val () = ((Engine.withModel model 64 (fn engine => (escaped:=SOME engine; raise ActionFailure));
           raise Fail "missing engine callback failure") handle ActionFailure => ())
val () = check (fails (fn () => Engine.useContext (valOf (!escaped)) (fn _ => ()))) "engine survived its scope"
val () = Engine.withModel model 64 (fn engine =>
    let val () = check (Engine.capacity engine=64) "wrong capacity"
        val () = check (fails (fn () => ignore (Engine.generate engine (Vector.fromList [~1]) 1))) "accepted invalid token"
        val first = Engine.generate engine (Vector.fromList [104]) 1
        val next = Engine.generate engine (#ids first) 1
    in check (#reused next=1) "cache replacement lost its prefix" end)
val () = print "Bulk row layouts, resource failure paths, options, all 65536 half bit patterns, narrowing, and engine lifetime checks passed\n"
