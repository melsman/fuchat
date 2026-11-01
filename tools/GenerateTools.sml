(* Generate typed dispatch from Futhark's manifest, not a runtime name lookup. *)
val manifest = Json.fromString (Util.readFile (hd (CommandLine.arguments ())))
val entries = case Json.get manifest "entry_points" of Json.Object entries => Json.sort entries
                                                    | _ => Util.fail "manifest lacks entry_points"
fun binding (name,definition) =
    if not (String.isPrefix "tool_" name) then NONE
    else
      case String.tokens (fn c => c = #"_") name of
        "tool"::short::description =>
          if (case Json.get definition "inputs" of
                Json.Array [input] => Json.get input "type" = Json.String "i64"
              | _ => false) andalso Json.get (Json.get definition "output") "type" = Json.String "i64" then
            SOME ("    {name=\"" ^ String.toString short ^ "\", description=\"calculate the "
                  ^ String.toString (String.concatWith " " description) ^ "\", invoke=qwen.Entry." ^ name ^ "}")
          else (Util.log ("Skipping unsupported tool signature: " ^ name); NONE)
      | _ => Util.fail ("invalid tool entry point name: " ^ name)
val bindings = List.mapPartial binding entries
val () = print "(* Generated from the Futhark manifest. *)\nstructure ToolEntries = struct\n\
               \  type entry = {name:string, description:string, invoke:qwen.ctx -> Int64.int -> Int64.int}\n\
               \  val all : entry list = [\n"
val () = print (String.concatWith ",\n" bindings ^ "\n  ]\nend\n")
