val ctx = qwen.Context.new qwen.Config.default

fun check () =
    let
        val result = qwen.Entry.tool_sumN1_sum_of_first_n_integers ctx
                         (Int64.fromInt 10)
        val () = if result = Int64.fromInt 55 then ()
                 else raise Fail "unexpected tool result"
        val cache = qwen.Entry.init ctx
                        (Int64.fromInt 1, Int64.fromInt 1,
                         Int64.fromInt 8, Int64.fromInt 128)
        val {cos, sin, kcache, vcache} = qwen.Opaque.Cache.values cache
        val () = if qwen.Real32Array4.shape kcache = (1, 1, 8, 128)
                 then () else raise Fail "unexpected cache shape"
        val () = qwen.Real32Array2.free cos
        val () = qwen.Real32Array2.free sin
        val () = qwen.Real32Array4.free kcache
        val () = qwen.Real32Array4.free vcache
        val () = qwen.Opaque.Cache.free cache
    in
        print "MLKit/Futhark integration check passed\n"
    end

val () = (check () handle ex => (qwen.Context.free ctx; raise ex))
val () = qwen.Context.free ctx
