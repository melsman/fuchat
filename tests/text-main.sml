val () = Protocol.lines (hd (CommandLine.arguments ())) (fn line =>
    case String.fields (fn c => c = #"\t") line of
      [flag,hex] =>
        let val bytes = Protocol.unhex hex
            val (text,tail) = Unicode.clean (flag="1") bytes
            val scalars = SOME (Unicode.scalars bytes) handle Chr => NONE
            val fields = case scalars of NONE => "invalid" | SOME values =>
                String.concatWith "," (Vector.foldr (fn ((cp,offset),xs) =>
                    (Int.toString cp ^ ":" ^ Int.toString offset)::xs) [] values)
        in print (Protocol.hex text ^ "\t" ^ Protocol.hex tail ^ "\t" ^ fields ^ "\n") end
    | _ => raise Fail "invalid Unicode test fixture")
