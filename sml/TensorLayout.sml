(* Convert source rows to contiguous destination runs. Bounds are checked by
   each destination slice, rather than by an Array.update for every element. *)
structure TensorLayout = struct
  datatype t = RowMajor | Transpose | Attention of int
  fun fill layout {rows,columns,first,count,offset} get store =
      case layout of
        RowMajor => store (offset+first*columns,count*columns,get)
      | Transpose => Util.for columns (fn column =>
          store (offset+column*rows+first,count,fn i => get (i*columns+column)))
      | Attention headDim =>
          let fun heads localRow =
                  if localRow=count then ()
                  else let val row = first+localRow
                           val head = row div headDim
                           val at = row mod headDim
                           val amount = Int.min(count-localRow,headDim-at)
                           val () = Util.for columns (fn column =>
                               store (offset+(head*columns+column)*headDim+at,amount,
                                      fn i => get ((localRow+i)*columns+column)))
                       in heads(localRow+amount) end
          in heads 0 end
end
