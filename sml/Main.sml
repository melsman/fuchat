structure Main = struct
  fun run arguments =
      let val options = Options.parse (if qwen.backend="cuda" then 1024 else 8192) arguments
      in case #mode options of Options.Chat => ChatApp.run options
                            | _ => Diagnostics.run options end
end
val () = Main.run (CommandLine.arguments ())
         handle ex => (Util.log ("fuchat: " ^ General.exnMessage ex); OS.Process.exit OS.Process.failure)
