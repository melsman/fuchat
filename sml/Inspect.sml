structure Inspect = struct
  fun weights params = Resource.scope (fn resources =>
      let
        val fields = qwen.Opaque.Params.values params
        (* Params.values returns independently owned projection handles. Register
           all of them before inspection, including fields not yet reported. *)
        val () = List.app (Resource.defer resources)
            [fn () => Matrices.A2.free (#token_embd fields),
             fn () => qwen.Real32Array1.free (#output_norm fields),
             fn () => Matrices.A2.free (#output fields),
             fn () => qwen.Real32Array2.free (#attn_norm fields),
             fn () => qwen.Real32Array2.free (#ffn_norm fields),
             fn () => Matrices.A3.free (#ffn_gate fields),
             fn () => Matrices.A3.free (#ffn_up fields),
             fn () => Matrices.A3.free (#ffn_down fields),
             fn () => Matrices.A4.free (#attn_q fields),
             fn () => Matrices.A4.free (#attn_k fields),
             fn () => Matrices.A4.free (#attn_v fields),
             fn () => Matrices.A3.free (#attn_output fields),
             fn () => qwen.Real32Array2.free (#q_norm fields),
             fn () => qwen.Real32Array2.free (#k_norm fields)]
        fun report name shape index =
              let val n = List.foldl (op *) 1 shape
                  val points = [0,Int.min (1,n-1),n div 3,n div 2,n-1]
                  val values = List.map (fn i => Real.toString (Real32.toLarge (index i))) points
              in print (name ^ "\t" ^ String.concatWith "," (List.map Int.toString shape) ^ "\t"
                        ^ String.concatWith "," values ^ "\n")
              end
        fun a1 name array =
            report name [qwen.Real32Array1.shape array] (fn i => qwen.Real32Array1.index array i)
        fun a2 name array =
            let val (a,b) = qwen.Real32Array2.shape array
            in report name [a,b] (fn i => qwen.Real32Array2.index array (i div b,i mod b))
            end
        fun m2 name array =
            let val (a,b) = Matrices.A2.shape array
            in report name [a,b] (fn i => Matrices.toReal32 (Matrices.A2.index array (i div b,i mod b)))
            end
        fun a3 name array =
            let val (a,b,c) = Matrices.A3.shape array
            in report name [a,b,c] (fn i => Matrices.toReal32 (Matrices.A3.index array (i div (b*c),i div c mod b,i mod c)))
            end
        fun a4 name array =
            let val (a,b,c,d) = Matrices.A4.shape array
            in report name [a,b,c,d] (fn i => Matrices.toReal32 (Matrices.A4.index array (i div (b*c*d),i div (c*d) mod b,i div d mod c,i mod d)))
            end
      in
        m2 "token_embd" (#token_embd fields); a1 "output_norm" (#output_norm fields);
        m2 "output" (#output fields); a2 "attn_norm" (#attn_norm fields);
        a2 "ffn_norm" (#ffn_norm fields); a3 "ffn_gate" (#ffn_gate fields);
        a3 "ffn_up" (#ffn_up fields); a3 "ffn_down" (#ffn_down fields);
        a4 "attn_q" (#attn_q fields); a4 "attn_k" (#attn_k fields); a4 "attn_v" (#attn_v fields);
        a3 "attn_output" (#attn_output fields); a2 "q_norm" (#q_norm fields); a2 "k_norm" (#k_norm fields)
      end)
end
