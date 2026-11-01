structure Engine = struct
  type t = {ctx:qwen.ctx, params:qwen.Opaque.Params.t,
            cache:qwen.Opaque.Cache.t option ref, tokens:int vector ref,
            size:int, vocab:int, eos:int, closed:bool ref}
  fun openModel model size : t =
      let
        val () = if size > 0 then () else Util.fail "context size must be positive"
        val dim = Weights.dimensions model
        val ctx = qwen.Context.new qwen.Config.default
        val params = Weights.load ctx model handle ex => (qwen.Context.free ctx; raise ex)
        val cache = qwen.Entry.init ctx
            (Int64.fromInt (#blocks dim),Int64.fromInt (#kvHeads dim),Int64.fromInt size,Int64.fromInt (#headDim dim))
            handle ex => (qwen.Opaque.Params.free params; qwen.Context.free ctx; raise ex)
      in {ctx=ctx,params=params,cache=ref (SOME cache),tokens=ref (Vector.fromList []),
          size=size,vocab= #vocab dim,eos=Gguf.integer model "tokenizer.ggml.eos_token_id",closed=ref false}
      end
  fun close (engine:t) =
      if !(#closed engine) then ()
      else (#closed engine := true;
            (case !(#cache engine) of NONE => () | SOME c => qwen.Opaque.Cache.free c);
            #cache engine := NONE; qwen.Opaque.Params.free (#params engine); qwen.Context.free (#ctx engine))
  fun clear (engine:t) = #tokens engine := Vector.fromList []
  fun prefix a b =
      let val n = Int.min (Vector.length a,Vector.length b)
          fun loop i = if i=n orelse Vector.sub(a,i) <> Vector.sub(b,i) then i else loop (i+1)
      in loop 0
      end
  fun generate (engine:t) ids requested =
      let
        val () = if !(#closed engine) then Util.fail "engine is closed" else ()
        val n = Vector.length ids
        val () = if n > 0 andalso n <= #size engine then () else Util.fail "prompt is empty or exceeds context size"
        val () = Vector.app (fn id => if id >= 0 andalso id < #vocab engine then () else Util.fail "token ID outside vocabulary") ids
        val reused = prefix (!(#tokens engine)) ids
        val count = Int.min (requested,#size engine - n + 1)
        val () = if count > 0 then () else Util.fail "generation count must be positive"
        (* The final generated token is not cached. Retain a nonempty suffix
           even if a caller supplies a prompt equal to the cached prefix. *)
        val reused = Int.min (reused,n-1)
        val input = Int64Array.tabulate (n-reused,fn i => Int64.fromInt (Vector.sub(ids,i+reused)))
        val xs = qwen.Int64Array1.new (#ctx engine) (Int64ArraySlice.full input) (n-reused)
        val cache = case !(#cache engine) of SOME c => c | NONE => Util.fail "cache unavailable"
        val result = qwen.Entry.gen (#ctx engine)
            (Int64.fromInt reused,xs,#params engine,cache,Int64.fromInt (#eos engine),Int64.fromInt count)
            handle ex => (qwen.Int64Array1.free xs; raise ex)
        val () = qwen.Int64Array1.free xs
        val (output,newCache) = qwen.Opaque.tup2_arr_i64_Cache.values result
        val () = qwen.Opaque.tup2_arr_i64_Cache.free result
        val () = qwen.Opaque.Cache.free cache
        val () = #cache engine := SOME newCache
        val values = Util.finally (fn () => qwen.Int64Array1.values output) (fn () => qwen.Int64Array1.free output)
        val out = Vector.tabulate (Int64Array.length values,fn i => Int64.toInt (Int64Array.sub(values,i)))
        val all = Vector.concat [ids,out]
        val cached = Int.max (0,Vector.length all - 1)
        val () = #tokens engine := Vector.tabulate (cached,fn i => Vector.sub(all,i))
      in {ids=all,newTokens=out,reused=reused}
      end
end
