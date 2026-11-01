(* Command-line configuration. Parsing has no I/O or process-exit effects. *)
signature OPTIONS = sig
  datatype mode = Chat | Help | Weights | Trace of string | Encode of string
                | Decode of string | TokenizerTest of string | RenderTest of string
                | Benchmark of string
  type t = {modelFile:string, context:int, count:int, limit:int,
            thinking:bool, tools:bool, maxToolRounds:int, system:string,
            prompt:string option, mode:mode, benchmarkCount:int, benchmarkRuns:int}
  val parse : int -> string list -> t
  val usage : string
end
