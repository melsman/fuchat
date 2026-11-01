(* Stage application sources per backend/precision. MLKit's per-source object
   caches must not be shared by projects importing different qwen structures.
   Basis and the independent Unicode package still use their normal caches. *)
val args = CommandLine.arguments ()
val root = List.nth(args,0)
val build = List.nth(args,1)
val precision = List.nth(args,2)
val template = List.nth(args,3)
val sourceDir = OS.Path.dir template
fun absolute path = OS.Path.mkAbsolute {path=path,relativeTo=root}
fun quoted path = "\"" ^ String.toString (absolute path) ^ "\"\n"
fun read path =
    let val input = TextIO.openIn path
        val text = TextIO.inputAll input handle ex => (TextIO.closeIn input; raise ex)
    in TextIO.closeIn input; text end
fun directory path =
    if OS.FileSys.isDir path handle OS.SysErr _ => false then ()
    else (directory (OS.Path.dir path); OS.FileSys.mkDir path)
fun stage relative =
    let
      val source = OS.Path.mkCanonical relative
      val target = OS.Path.concat(OS.Path.concat(build,"sources"),source)
      val text = read (absolute source)
      val old = SOME (read (absolute target)) handle IO.Io _ => NONE
      val () = if old=SOME text then ()
               else let val () = directory (absolute (OS.Path.dir target))
                        val output = TextIO.openOut (absolute target)
                    in TextIO.output(output,text); TextIO.closeOut output end
    in target end
val stream = TextIO.openIn template
fun process line =
    let val path = String.concat (String.tokens Char.isSpace line)
    in
      if path="" then print "\n"
      else if String.isPrefix "$(SML_LIB)" path then print (path ^ "\n")
      else if String.isPrefix "../build/" path then
        (print (quoted (OS.Path.concat(build,String.extract(path,9,NONE))));
         if path="../build/qwen.sml" then print (quoted (stage ("sml/precision/" ^ precision ^ ".sml"))) else ())
      else if String.isSuffix ".sml" path then print (quoted (stage (OS.Path.concat(sourceDir,path))))
      else print (quoted (OS.Path.concat(sourceDir,path)))
    end
fun loop () = case TextIO.inputLine stream of NONE => () | SOME line => (process line; loop ())
val () = (loop () handle ex => (TextIO.closeIn stream; raise ex))
val () = TextIO.closeIn stream
