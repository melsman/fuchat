structure Weights :> WEIGHTS = struct
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
        val () = if List.all (fn n => n>0) [#blocks dim,#vocab dim,#embedding dim,#feedForward dim,#heads dim,#kvHeads dim]
                 andalso #headDim dim = 128 andalso #heads dim mod #kvHeads dim = 0 then ()
                 else Util.fail "the Futhark engine requires head dimension 128 and divisible head counts"
      in dim
      end

  fun load ctx model =
      let
        (* Retain ownership of the return handle until temporary cleanup and
           closing the reader have both succeeded. *)
        val result = ref (NONE:qwen.Opaque.Params.t option)
        fun prepare r = Resource.scope (fn resources =>
          let
            val {blocks=b,vocab=v,embedding=d,feedForward=f,heads=h,kvHeads=kh,headDim=dh} = dimensions model
            fun own free array = Resource.own resources free array
            fun field allocate fill zero readRows names expected layout =
                let
                  val tensors = List.map (Gguf.tensor model) names
                  val () = List.app (fn (t:Gguf.tensor) =>
                      if #shape t = expected then () else Util.fail ("unexpected tensor shape: " ^ #name t)) tensors
                  val perBlock = List.foldl (op *) 1 expected
                  val columns = List.last expected
                  val rows = perBlock div columns
                  val out = allocate (List.length tensors*perBlock, zero)
                  val () = Util.log ("Loading " ^ hd names)
                  fun read ([],_) = ()
                    | read (t::rest,i) =
                        (readRows model r t columns (fn (first,count,get) =>
                            TensorLayout.fill layout {rows=rows,columns=columns,first=first,count=count,offset=i*perBlock}
                                              get (fill out)); read (rest,i+1))
                  val () = read (tensors,0)
                in out
                end
            fun matrix names expected layout = field Matrices.Array.array Matrices.fill Matrices.zero Matrices.readRows names expected layout
            fun fill32 array (start,count,get) =
                Real32ArraySlice.modifyi (fn (i,_) => get i) (Real32ArraySlice.slice(array,start,SOME count))
            fun normal names expected = field Real32Array.array fill32 (Real32.fromInt 0) Gguf.readTensor32Rows names expected TensorLayout.RowMajor
            fun single name expected layout = matrix [name ^ ".weight"] expected layout
            fun blockNames name = List.tabulate (b,fn i => "blk." ^ Int.toString i ^ "." ^ name ^ ".weight")
            fun block name expected layout = matrix (blockNames name) expected layout
            fun a1 data n = own qwen.Real32Array1.free (qwen.Real32Array1.new ctx (Real32ArraySlice.full data) n)
            fun a2 data shape = own qwen.Real32Array2.free (qwen.Real32Array2.new ctx (Real32ArraySlice.full data) shape)
            fun a3 data shape = own Matrices.A3.free (Matrices.A3.new ctx (Matrices.Slice.full data) shape)
            fun a4 data shape = own Matrices.A4.free (Matrices.A4.new ctx (Matrices.Slice.full data) shape)
            fun m2 data shape = own Matrices.A2.free (Matrices.A2.new ctx (Matrices.Slice.full data) shape)
            fun build () =
                let
                  val token_embd = m2 (single "token_embd" [v,d] TensorLayout.RowMajor) (v,d)
                  val output_norm = a1 (normal ["output_norm.weight"] [d]) d
                  val output = m2 (single "output" [v,d] TensorLayout.Transpose) (d,v)
                  val attn_norm = a2 (normal (blockNames "attn_norm") [d]) (b,d)
                  val ffn_norm = a2 (normal (blockNames "ffn_norm") [d]) (b,d)
                  val ffn_gate = a3 (block "ffn_gate" [f,d] TensorLayout.Transpose) (b,d,f)
                  val ffn_up = a3 (block "ffn_up" [f,d] TensorLayout.Transpose) (b,d,f)
                  val ffn_down = a3 (block "ffn_down" [d,f] TensorLayout.Transpose) (b,f,d)
                  val attn_q = a4 (block "attn_q" [h*dh,d] (TensorLayout.Attention dh)) (b,h,d,dh)
                  val attn_k = a4 (block "attn_k" [kh*dh,d] (TensorLayout.Attention dh)) (b,kh,d,dh)
                  val attn_v = a4 (block "attn_v" [kh*dh,d] (TensorLayout.Attention dh)) (b,kh,d,dh)
                  val attn_output = a3 (block "attn_output" [d,h*dh] TensorLayout.Transpose) (b,h*dh,d)
                  val q_norm = a2 (normal (blockNames "attn_q_norm") [dh]) (b,dh)
                  val k_norm = a2 (normal (blockNames "attn_k_norm") [dh]) (b,dh)
                  val params = qwen.Opaque.Params.new ctx
                     {token_embd=token_embd,output_norm=output_norm,output=output,
                      attn_norm=attn_norm,ffn_norm=ffn_norm,ffn_gate=ffn_gate,ffn_up=ffn_up,
                      ffn_down=ffn_down,attn_q=attn_q,attn_k=attn_k,attn_v=attn_v,
                      attn_output=attn_output,q_norm=q_norm,k_norm=k_norm}
                  val () = result := SOME params
                in params end
          in build ()
          end)
      in Gguf.withReader model prepare
         handle ex => ((case !result of NONE => () | SOME params =>
                         (qwen.Opaque.Params.free params handle _ => ()));
                       raise ex)
      end
end
