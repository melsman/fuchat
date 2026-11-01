(* Shared startup for the chat and diagnostic front ends. *)
structure Host = struct
  fun withModel (options:Options.t) action =
      (Util.log ("Backend: " ^ qwen.backend ^ ", precision: " ^ Matrices.precision);
       action (Gguf.load (#modelFile options)))
  fun withEngine (options:Options.t) model action =
      let val start = Time.now ()
      in Engine.withModel model (#context options) (fn engine =>
           action (engine,Time.toReal (Time.-(Time.now (),start)))) end
end
