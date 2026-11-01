structure Weights = struct
  type dimensions = {blocks:int, vocab:int, embedding:int, feedForward:int,
                     heads:int, kvHeads:int, headDim:int}
  fun dimensions model : dimensions =
      let
        val () = if Gguf.string model "general.architecture" = "qwen3" then () else Util.fail "only Qwen3 is supported"
        val dim = {blocks=Gguf.integer model "qwen3.block_count",
                   vocab=Vector.length (Gguf.strings model "tokenizer.ggml.tokens"),
                   embedding=Gguf.integer model "qwen3.embedding_length",
                   feedForward=Gguf.integer model "qwen3.feed_forward_length",
                   heads=Gguf.integer model "qwen3.attention.head_count",
                   kvHeads=Gguf.integer model "qwen3.attention.head_count_kv",
                   headDim=Gguf.integer model "qwen3.attention.key_length"}
        val () = if #headDim dim = 128 andalso #heads dim mod #kvHeads dim = 0 then ()
                 else Util.fail "the Futhark engine requires head dimension 128 and divisible head counts"
      in dim
      end

  fun load ctx model =
      let
        val {blocks=b,vocab=v,embedding=d,feedForward=f,heads=h,kvHeads=kh,headDim=dh} = dimensions model
        val r = Gguf.openReader (#path model)
        val pending = ref ([] : (unit -> unit) list)
        fun cleanup () = List.app (fn free => free ()) (!pending)
        fun own free array = (pending := (fn () => free array) :: !pending; array)
        fun field names expected permute =
            let
              val tensors = List.map (Gguf.tensor model) names
              val () = List.app (fn (t:Gguf.tensor) =>
                  if #shape t = expected then () else Util.fail ("unexpected tensor shape: " ^ #name t)) tensors
              val perBlock = List.foldl (op *) 1 expected
              val out = Real32Array.array (List.length tensors*perBlock, Real32.fromInt 0)
              val () = Util.log ("Loading " ^ hd names)
              fun read ([],_) = ()
                | read (t::rest,i) =
                    (Gguf.readTensor model r t out (fn j => i*perBlock + permute j); read (rest,i+1))
              val () = read (tensors,0)
            in out
            end
        fun single name expected permute = field [name ^ ".weight"] expected permute
        fun blockNames name = List.tabulate (b,fn i => "blk." ^ Int.toString i ^ "." ^ name ^ ".weight")
        fun block name expected permute = field (blockNames name) expected permute
        fun transpose columns rows i = (i mod columns)*rows + i div columns
        fun attention heads i =
            let val row = i div d
            in (row div dh)*d*dh + (i mod d)*dh + row mod dh
            end
        fun a1 data n = own qwen.Real32Array1.free (qwen.Real32Array1.new ctx (Real32ArraySlice.full data) n)
        fun a2 data shape = own qwen.Real32Array2.free (qwen.Real32Array2.new ctx (Real32ArraySlice.full data) shape)
        fun a3 data shape = own qwen.Real32Array3.free (qwen.Real32Array3.new ctx (Real32ArraySlice.full data) shape)
        fun a4 data shape = own qwen.Real32Array4.free (qwen.Real32Array4.new ctx (Real32ArraySlice.full data) shape)
        fun build () =
            let
              val token_embd = a2 (single "token_embd" [v,d] (fn i => i)) (v,d)
              val output_norm = a1 (single "output_norm" [d] (fn i => i)) d
              val output = a2 (single "output" [v,d] (transpose d v)) (d,v)
              val attn_norm = a2 (block "attn_norm" [d] (fn i => i)) (b,d)
              val ffn_norm = a2 (block "ffn_norm" [d] (fn i => i)) (b,d)
              val ffn_gate = a3 (block "ffn_gate" [f,d] (transpose d f)) (b,d,f)
              val ffn_up = a3 (block "ffn_up" [f,d] (transpose d f)) (b,d,f)
              val ffn_down = a3 (block "ffn_down" [d,f] (transpose f d)) (b,f,d)
              val attn_q = a4 (block "attn_q" [h*dh,d] (attention h)) (b,h,d,dh)
              val attn_k = a4 (block "attn_k" [kh*dh,d] (attention kh)) (b,kh,d,dh)
              val attn_v = a4 (block "attn_v" [kh*dh,d] (attention kh)) (b,kh,d,dh)
              val attn_output = a3 (block "attn_output" [d,h*dh] (transpose (h*dh) d)) (b,h*dh,d)
              val q_norm = a2 (block "attn_q_norm" [dh] (fn i => i)) (b,dh)
              val k_norm = a2 (block "attn_k_norm" [dh] (fn i => i)) (b,dh)
            in qwen.Opaque.Params.new ctx
                 {token_embd=token_embd,output_norm=output_norm,output=output,
                  attn_norm=attn_norm,ffn_norm=ffn_norm,ffn_gate=ffn_gate,ffn_up=ffn_up,
                  ffn_down=ffn_down,attn_q=attn_q,attn_k=attn_k,attn_v=attn_v,
                  attn_output=attn_output,q_norm=q_norm,k_norm=k_norm}
            end
        fun release () = (cleanup (); Gguf.close r)
      in Util.finally build release
      end
end
