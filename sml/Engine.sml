structure Engine :> ENGINE = struct
  type result = {ids:int vector, newTokens:int vector, reused:int}
  type t = {ctx:qwen.ctx, params:qwen.Opaque.Params.t, resources:Resource.t,
            cache:qwen.Opaque.Cache.t option ref, tokens:int vector ref,
            size:int, vocab:int, eos:int, closed:bool ref}
  fun openModel model size : t =
      let
        val () = if size > 0 then () else Util.fail "context size must be positive"
        val dim = Weights.dimensions model
        val eos = Gguf.integer model "tokenizer.ggml.eos_token_id"
        val () = if eos>=0 andalso eos< #vocab dim then () else Util.fail "EOS token outside vocabulary"
        val resources = Resource.create ()
        fun build () =
            let
              val ctx = Resource.own resources qwen.Context.free (qwen.Context.new (Matrices.config ()))
              fun prepare () =
                  let
                    val params = Resource.own resources qwen.Opaque.Params.free (Weights.load ctx model)
                    val cache = ref (SOME (qwen.Entry.init ctx
                        (Int64.fromInt (#blocks dim),Int64.fromInt (#kvHeads dim),
                         Int64.fromInt size,Int64.fromInt (#headDim dim))))
                    val () = Resource.defer resources (fn () =>
                        case !cache of NONE => () | SOME value => (cache:=NONE; qwen.Opaque.Cache.free value))
                  in {ctx=ctx,params=params,resources=resources,cache=cache,tokens=ref (Vector.fromList []),
                      size=size,vocab= #vocab dim,eos=eos,closed=ref false} end
            in prepare () handle ex => ((qwen.Context.sync ctx handle _ => ()); raise ex) end
      in Resource.onFailure resources build end
  fun close (engine:t) =
      if !(#closed engine) then ()
      else (#closed engine := true;
            Util.finally (fn () => qwen.Context.sync (#ctx engine))
                         (fn () => Resource.release (#resources engine)))
  fun withModel model size action =
      let val engine = openModel model size
      in Util.finally (fn () => action engine) (fn () => close engine) end
  fun requireOpen (engine:t) = if !(#closed engine) then Util.fail "engine is closed" else ()
  fun capacity (engine:t) = #size engine
  fun eos (engine:t) = #eos engine
  fun useContext (engine:t) action = (requireOpen engine; action (#ctx engine))
  fun useParams (engine:t) action = (requireOpen engine; action (#params engine))
  fun clear (engine:t) = (requireOpen engine; #tokens engine := Vector.fromList [])
  fun prefix a b =
      let val n = Int.min (Vector.length a,Vector.length b)
          fun loop i = if i=n orelse Vector.sub(a,i) <> Vector.sub(b,i) then i else loop (i+1)
      in loop 0 end
  fun generate (engine:t) ids requested =
      let
        val () = requireOpen engine
        val n = Vector.length ids
        val () = if n > 0 andalso n <= #size engine then () else Util.fail "prompt is empty or exceeds context size"
        val () = Vector.app (fn id => if id >= 0 andalso id < #vocab engine then () else Util.fail "token ID outside vocabulary") ids
        val count = Int.min (requested,#size engine - n + 1)
        val () = if count > 0 then () else Util.fail "generation count must be positive"
        (* The final output token is not cached. Keep a nonempty suffix even
           when the input equals the cached prefix. *)
        val reused = Int.min (prefix (!(#tokens engine)) ids,n-1)
        fun run resources =
            let
              val input = Int64Array.tabulate (n-reused,fn i => Int64.fromInt (Vector.sub(ids,i+reused)))
              val xs = Resource.own resources qwen.Int64Array1.free
                  (qwen.Int64Array1.new (#ctx engine) (Int64ArraySlice.full input) (n-reused))
              val cache = case !(#cache engine) of SOME c => c | NONE => Util.fail "cache unavailable"
              val result = Resource.own resources qwen.Opaque.tup2_arr_i64_Cache.free
                  (qwen.Entry.gen (#ctx engine)
                    (Int64.fromInt reused,xs,#params engine,cache,Int64.fromInt (#eos engine),Int64.fromInt count))
              (* Projection returns independently owned handles. Keep both
                 scoped until output extraction and host allocation succeed. *)
              val (output,newCache) = qwen.Opaque.tup2_arr_i64_Cache.values result
              val () = Resource.defer resources (fn () => qwen.Int64Array1.free output)
              val pending = ref (SOME newCache)
              val () = Resource.defer resources (fn () =>
                  case !pending of NONE => () | SOME c => qwen.Opaque.Cache.free c)
              val values = qwen.Int64Array1.values output
              val out = Vector.tabulate (Int64Array.length values,fn i => Int64.toInt (Int64Array.sub(values,i)))
              val all = Vector.concat [ids,out]
              val cached = Vector.tabulate (Int.max (0,Vector.length all-1),fn i => Vector.sub(all,i))
              val () = #cache engine := NONE
              val () = qwen.Opaque.Cache.free cache
              val () = #cache engine := SOME newCache
              val () = pending := NONE (* Transfer ownership to the engine. *)
              val () = #tokens engine := cached
            in {ids=all,newTokens=out,reused=reused} end
      in Resource.scope (fn resources => run resources
           handle ex => ((qwen.Context.sync (#ctx engine) handle _ => ()); raise ex))
         handle ex => ((close engine handle _ => ()); raise ex)
      end
end
